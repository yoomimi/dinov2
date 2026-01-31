#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv2 patch feature precompute (224x224) -> scene별 HDF5 저장
- RAM 12GB 환경을 전제로 "스트리밍 + 작은 배치 + scene 단위 파일"로 안정성 우선
- GPU당 프로세스 1개(spawn), scene는 round-robin 샤딩
- 결과: {output_dir}/{scene}.h5  안에 /features (N, C, 16, 16) fp16 저장

필수 설치:
  pip install h5py pillow tqdm

주의:
- 같은 scene의 h5를 여러 프로세스가 동시에 쓰지 않도록 설계(샤딩)
- torch.hub로 dinov2 로딩 (환경에 따라 첫 실행 시 모델 다운로드 필요)
"""

import os
import re
import argparse
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import torch
import h5py
from PIL import Image
from tqdm import tqdm


# -------------------------
# Utils
# -------------------------

def list_scene_dirs(input_root_dir: str) -> List[str]:
    scenes = [
        d for d in os.listdir(input_root_dir)
        if os.path.isdir(os.path.join(input_root_dir, d))
    ]
    scenes.sort()
    return scenes


def shard_list(items: List[str], num_shards: int, shard_id: int) -> List[str]:
    return items[shard_id::num_shards]


def list_jpgs(scene_dir: str) -> List[str]:
    files = [f for f in os.listdir(scene_dir) if f.lower().endswith(".jpg")]
    files.sort()
    return files


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_rename(tmp_path: str, final_path: str):
    os.replace(tmp_path, final_path)


# DINOv2 표준 정규화(Imagenet 계열 mean/std)
# (DINOv2 공식도 동일 mean/std 사용)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def preprocess_pil_to_tensor_224(pil_img: Image.Image) -> torch.Tensor:
    """
    224x224로 resize(강제) + float tensor + normalize
    - RAM 안정성을 위해 numpy intermediate 최소화
    - 반환: (3,224,224) float32 tensor (CPU)
    """
    img = pil_img.convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)  # (224,224,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t


def infer_patch_grid(tokens: torch.Tensor) -> Tuple[int, int]:
    """
    tokens: (B, N, C) where N is number of patch tokens (no cls/register)
    224 with patch14 => N=256 => 16x16
    일반화: N이 완전제곱이면 sqrt 사용
    """
    n = tokens.shape[1]
    s = int(round(n ** 0.5))
    if s * s != n:
        raise ValueError(f"Patch token count {n} is not a perfect square; cannot reshape to grid.")
    return s, s


def load_dinov2_backbone(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    torch.hub DINOv2 로드.
    예시 model_name:
      - dinov2_vits14
      - dinov2_vitb14
      - dinov2_vitl14
      - dinov2_vitg14
    registers를 쓰는 variant가 hub에 있는 환경도 있고 없는 환경도 있어,
    여기서는 backbone만(패치 토큰) 뽑는 목적이므로 일반 모델로 충분.
    """
    # torch.hub repo는 보통 'facebookresearch/dinov2'
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def load_dinov2_reg4_vitl14(ckpt_path: str, device: torch.device):
    import dinov2.models.vision_transformer as vits

    model = vits.vit_large(
        patch_size=14,
        num_register_tokens=4,
        img_size=224,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_patch_features(
    model: torch.nn.Module,
    batch_imgs: torch.Tensor,
    amp: bool = True
) -> torch.Tensor:
    """
    batch_imgs: (B,3,224,224) float32 on GPU
    return: (B,C,H',W') fp16 on GPU (H'=W'=16 for 224/14)
    """
    # dinov2 hub 모델은 forward_features 제공
    # out["x_norm_patchtokens"]: (B, N, C)
    with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
        out = model.forward_features(batch_imgs)
        patch = out["x_norm_patchtokens"]  # (B, N, C)
        gh, gw = infer_patch_grid(patch)
        # (B, N, C) -> (B, gh, gw, C) -> (B, C, gh, gw)
        patch = patch.view(patch.shape[0], gh, gw, patch.shape[-1]).permute(0, 3, 1, 2).contiguous()
        # 저장은 fp16 고정
        return patch.to(torch.float16)


# -------------------------
# HDF5 writer
# -------------------------

def create_h5_features(
    h5_path_tmp: str,
    n_frames: int,
    c: int,
    chunk1: bool = True,
    compression: str = "lzf",
):
    """
    (N, C, 16, 16) fp16 dataset 생성.
    chunk1=True면 프레임 단위 접근 최적: (1, C, 16, 16)
    """
    ensure_dir(os.path.dirname(h5_path_tmp))
    h5 = h5py.File(h5_path_tmp, "w")

    chunks = (1, c, 16, 16) if chunk1 else None
    # gzip은 CPU/시간 부담이 커서 RAM 작은 환경에서는 lzf가 무난
    # compression=None 도 가능
    ds = h5.create_dataset(
        "features",
        shape=(n_frames, c, 16, 16),
        dtype=np.float16,
        chunks=chunks,
        compression=compression if compression and compression.lower() != "none" else None,
        shuffle=True if compression and compression.lower() in ("gzip", "lzf") else False,
    )
    return h5, ds


def write_meta(h5: h5py.File, model_name: str):
    g = h5.create_group("meta")
    g.attrs["model"] = "dinov2_vitl14_reg4"
    g.attrs["num_register_tokens"] = 4
    g.attrs["input_resolution_hw"] = (224, 224)
    g.attrs["patch_size"] = 14
    g.attrs["token_grid_hw"] = (16, 16)
    g.attrs["dtype"] = "float16"
    g.attrs["layout"] = "(N, C, 16, 16)"


# -------------------------
# Worker
# -------------------------

def worker_run(
    shard_id: int,
    num_shards: int,
    input_root_dir: str,
    output_root_dir: str,
    model_name: str,
    batch_size: int,
    compression: str,
    skipped_scenes,
    failed_scenes,
):
    # GPU 설정
    torch.cuda.set_device(shard_id)
    device = torch.device("cuda", shard_id)

    # 모델 로드(프로세스당 1회)
    model = load_dinov2_reg4_vitl14(
        ckpt_path="/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
        device=device,
    )
    # scene 샤딩
    scenes = shard_list(list_scene_dirs(input_root_dir), num_shards=num_shards, shard_id=shard_id)

    pbar_scenes = tqdm(scenes, desc=f"[GPU{shard_id}] scenes", position=shard_id, leave=True)

    for scene in pbar_scenes:
        scene_in = os.path.join(input_root_dir, scene)
        scene_out = os.path.join(output_root_dir, f"{scene}.h5")
        scene_tmp = os.path.join(output_root_dir, f".{scene}.tmp.h5")

        try:
            # 이미 완성 파일 있으면 skip (resume)
            if os.path.exists(scene_out):
                continue

            jpgs = list_jpgs(scene_in)
            if not jpgs:
                skipped_scenes.append(scene)
                pbar_scenes.write(f"[GPU{shard_id}] skip(empty) {scene}")
                continue

            # 첫 배치로 C 확정
            # (모델에 따라 C가 다를 수 있어, 한번 forward로 확인)
            first_paths = [os.path.join(scene_in, jpgs[0])]
            pil = Image.open(first_paths[0])
            t = preprocess_pil_to_tensor_224(pil).unsqueeze(0).to(device, non_blocking=True)
            feat = extract_patch_features(model, t, amp=True)  # (1,C,16,16)
            c = int(feat.shape[1])
            del t, feat
            torch.cuda.empty_cache()

            # H5 생성(atomic write를 위해 tmp로 작성 후 rename)
            h5, ds = create_h5_features(
                h5_path_tmp=scene_tmp,
                n_frames=len(jpgs),
                c=c,
                chunk1=True,
                compression=compression,
            )
            write_meta(h5, model_name=model_name)

            # frame id를 파일명에서 뽑고 싶으면 여기서 저장 가능(선택)
            # 예: 000123.jpg -> 123
            # ids = np.array([int(re.findall(r"\d+", f)[-1]) if re.findall(r"\d+", f) else i for i,f in enumerate(jpgs)], dtype=np.int32)
            # h5.create_dataset("frame_ids", data=ids)

            pbar_imgs = tqdm(
                range(0, len(jpgs), batch_size),
                desc=f"[GPU{shard_id}] {scene}",
                position=shard_id + num_shards,
                leave=False,
            )

            for start in pbar_imgs:
                end = min(start + batch_size, len(jpgs))
                batch_files = jpgs[start:end]

                # CPU에서 배치 텐서 구성 (RAM 절약: 한 배치씩만)
                batch_cpu = []
                for f in batch_files:
                    path = os.path.join(scene_in, f)
                    with Image.open(path) as im:
                        batch_cpu.append(preprocess_pil_to_tensor_224(im))
                batch_cpu = torch.stack(batch_cpu, dim=0)  # (B,3,224,224) float32 CPU

                # GPU로 이동 후 feature 추출
                batch_gpu = batch_cpu.to(device, non_blocking=True)
                feats_gpu = extract_patch_features(model, batch_gpu, amp=True)  # (B,C,16,16) fp16

                # CPU로 내려서 바로 기록 (RAM 절약: numpy로 바로)
                feats_np = feats_gpu.detach().cpu().numpy()  # float16

                # H5에 기록
                ds[start:end, :, :, :] = feats_np

                # 메모리 해제
                del batch_cpu, batch_gpu, feats_gpu, feats_np

            h5.flush()
            h5.close()

            # atomic rename
            atomic_rename(scene_tmp, scene_out)

        except Exception as e:
            # 실패 시 tmp 제거(깨진 파일 방지)
            try:
                if os.path.exists(scene_tmp):
                    os.remove(scene_tmp)
            except Exception:
                pass
            failed_scenes.append(scene)
            pbar_scenes.write(f"[GPU{shard_id}] FAIL {scene}: {repr(e)}")
            continue

    # 정리
    del model
    torch.cuda.empty_cache()


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="scene 디렉토리들이 들어있는 루트")
    parser.add_argument("--output_dir", type=str, required=True, help="scene별 feature h5 저장 루트")
    parser.add_argument("--num_gpus", type=int, default=4, help="사용 GPU 수")
    parser.add_argument("--model", type=str, default="dinov2_vitl14", help="torch.hub dinov2 모델명")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU당 배치 (RAM 12GB면 8~16 추천)")
    parser.add_argument("--compression", type=str, default="lzf", help="h5 압축: lzf|gzip|none")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    available = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    skipped_scenes = manager.list()
    failed_scenes = manager.list()

    procs = []
    for shard_id in range(num_gpus):
        p = ctx.Process(
            target=worker_run,
            args=(
                shard_id,
                num_gpus,
                args.input_dir,
                args.output_dir,
                args.model,
                args.batch_size,
                args.compression,
                skipped_scenes,
                failed_scenes,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # exit code 체크
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"One or more workers crashed. exitcodes={bad}")

    # 요약 출력
    skipped = sorted(set(skipped_scenes))
    failed = sorted(set(failed_scenes))

    print("\n========== SUMMARY ==========")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    if skipped:
        print(f"Skipped scenes ({len(skipped)}):")
        for s in skipped[:50]:
            print(f" - {s}")
        if len(skipped) > 50:
            print(f" ... and {len(skipped)-50} more")
    else:
        print("Skipped scenes: none")

    if failed:
        print(f"Failed scenes ({len(failed)}):")
        for s in failed[:50]:
            print(f" - {s}")
        if len(failed) > 50:
            print(f" ... and {len(failed)-50} more")
    else:
        print("Failed scenes: none")
    print("=============================\n")


if __name__ == "__main__":
    main()
