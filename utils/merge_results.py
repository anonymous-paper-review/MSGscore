# VBench + Eval

import os
import json
import re
import shutil
import argparse
from glob import glob
import pandas as pd


def list_subfolders(folder_path):
    subfolders = [name for name in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, name))]
    return subfolders

def parse_score_file(filepath, pattern):
    scores = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                filename = match.group(1)
                try:
                    score = float(match.group(2))
                    scores[filename] = score
                except ValueError:
                    continue
    return scores

def merge_benchmarks(folder):
    # Step 1: Load benchmark JSON
    json_files = glob(os.path.join(folder, "eval_*.json"))
    if not json_files:
        print(f"❌ {folder} - benchmark JSON 파일 없음")
        return
    with open(json_files[0], "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Step 1.1: dynamic_degree 값을 True/False에서 0.9/0.1으로 변경
    if "dynamic_degree" in benchmark_data:
        if isinstance(benchmark_data["dynamic_degree"], list) and len(benchmark_data["dynamic_degree"]) > 1:
            for entry in benchmark_data["dynamic_degree"][1]:
                if "video_results" in entry and isinstance(entry["video_results"], bool):
                    entry["video_results"] = 0.9 if entry["video_results"] else 0.1

    # Step 2: 모든 metric 파일 정의
    metric_files = {
        "inception_score": ("IS_record.txt", r".*/(\d+\.mp4):\s+([\d\.eE\+\-]+)"),
        "face_consistency_score": ("face_consistency_score_record.txt", r"Vid: (\d+\.mp4),\s+Current face_consistency_score: ([\d\.]+)"),
        "sd_score": ("sd_score_record.txt", r"Vid: (\d+\.mp4),\s+Current sd_score: ([\d\.]+)"),
        "flow_score": ("flow_score_record.txt", r"Vid: (\d+\.mp4),\s+Current flow_score: ([\d\.eE\+\-]+)"),
        "warping_error": ("warping_error_record.txt", r"Vid: (\d+\.mp4),\s+Current warping_error: ([\d\.eE\-]+)"),
        "blip_bleu": ("blip_bleu_record.txt", r"Vid: (\d+\.mp4),\s+Current blip_bleu: ([\d\.eE\+\-]+)"),
        "clip_score": ("clip_score_record.txt", r"Vid: (\d+\.mp4),\s+Current clip_score: ([\d\.eE\+\-]+)"),
        "clip_temp_score": ("clip_temp_score_record.txt", r"Vid: (\d+\.mp4),\s+Current clip_temp_score: ([\d\.eE\+\-]+)")
    }

    # Step 3: 각 metric 파일 파싱
    all_scores = {}
    for metric, (filename, pattern) in metric_files.items():
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            parsed = parse_score_file(path, pattern)
            for fname, score in parsed.items():
                all_scores.setdefault(fname, {})[metric] = score

    # Step 4: dover.csv (뒤에서부터 3열 기준 + 예외 줄 무시)
    dover_path = os.path.join(folder, "dover.csv")
    if os.path.exists(dover_path):
        bad_lines = []
        clean_rows = []

        with open(dover_path, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header
            for i, line in enumerate(f, start=2):  # 2부터 시작 (줄 번호)
                line = line.strip()
                if not line or line.startswith("Avg."):
                    continue  # 평균 줄은 건너뜀

                parts = line.rsplit(",", 3)
                if len(parts) < 4:
                    bad_lines.append((i, line))
                    continue

                path = parts[0].strip()
                aesthetic = parts[1].strip()
                technical = parts[2].strip()

                try:
                    filename = f"{path}.mp4"
                    all_scores.setdefault(filename, {})["aesthetic_score"] = float(aesthetic)
                    all_scores[filename]["technical_score"] = float(technical)
                except ValueError:
                    bad_lines.append((i, line))

        if bad_lines:
            print("⚠️ dover.csv에서 파싱 실패한 줄:")
            for lineno, content in bad_lines:
                print(f"  Line {lineno}: {content}")

    # Step 5: 모든 metric 이름 (benchmark + 외부)
    benchmark_metrics = list(benchmark_data.keys())
    external_metrics = list(metric_files.keys()) + ["aesthetic_score", "technical_score"]
    required_metrics = benchmark_metrics + external_metrics

    # Step 6: 병합 (모든 metric 다 있을 때만 포함)
    merged = {}

    for metric, data in benchmark_data.items():
        if isinstance(data, list) and len(data) > 1:
            for entry in data[1]:
                path = entry.get("video_path")
                if not path:
                    continue
                filename = os.path.basename(path)
                if filename not in all_scores:
                    continue

                # 임시 dict 생성
                temp = {metric: entry["video_results"]}
                for other_metric in benchmark_metrics:
                    if other_metric == metric:
                        continue
                    for other_entry in benchmark_data[other_metric][1]:
                        if os.path.basename(other_entry.get("video_path", "")) == filename:
                            temp[other_metric] = other_entry.get("video_results")
                            break

                # 외부 metric 추가
                for m in external_metrics:
                    if m in all_scores[filename]:
                        temp[m] = all_scores[filename][m]

                # ✅ 누락된 metric 검사
                missing = [k for k in required_metrics if k not in temp]
                if missing:
                    print(f"⚠️ {filename} 누락된 metric: {missing}")
                else:
                    merged[filename] = temp

    # Step 7: 저장
    output_path = os.path.join(folder, "merged_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"✅ 병합 완료: {output_path} | 포함된 영상 수: {len(merged)}")


def find_timestamp_folder(base_path, timestamp):
    """주어진 timestamp가 포함된 폴더를 찾습니다."""
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and timestamp in item:
            return item_path
    return None


def find_timestamp_files(base_path, timestamp):
    """주어진 timestamp가 포함된 파일 목록을 찾습니다."""
    matching_files = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isfile(item_path) and timestamp in item:
            matching_files.append(item_path)
    return matching_files


def main():
    parser = argparse.ArgumentParser(description="MSGscore 병합 도구")
    parser.add_argument("--ec_path", required=True, help="MSGscore 경로")
    parser.add_argument("--prompt_dir", help="프롬프트 디렉토리 경로")
    parser.add_argument("--output_dir", help="출력 디렉토리 경로")
    parser.add_argument("--timestamp", required=True, help="타임스탬프")
    
    args = parser.parse_args()
    
    # 1. MSGscore/EvalCrafter/result에서 timestamp와 일치하는 폴더 찾기
    eval_path = os.path.join(args.ec_path, "EvalCrafter", "result")
    eval_folder = find_timestamp_folder(eval_path, args.timestamp)
    
    if not eval_folder:
        print(f"❌ 타임스탬프 {args.timestamp}와 일치하는 폴더를 찾을 수 없습니다: {eval_path}")
        return
    
    print(f"✅ 찾은 EvalCrafter 결과 폴더: {eval_folder}")
    
    # 2. MSGscore/result로 폴더 복사
    result_path = os.path.join(args.ec_path, "result")
    os.makedirs(result_path, exist_ok=True)
    
    folder_name = os.path.basename(eval_folder)
    target_folder = os.path.join(result_path, folder_name)
    
    if os.path.exists(target_folder):
        print(f"⚠️ 이미 존재하는 폴더입니다: {target_folder}")
    else:
        shutil.copytree(eval_folder, target_folder)
        print(f"✅ 폴더 복사 완료: {target_folder}")
    
    # 3. MSGscore/Vench/result에서 timestamp와 일치하는 파일 찾기
    vbench_path = os.path.join(args.ec_path, "VBench", "result")
    vbench_files = find_timestamp_files(vbench_path, args.timestamp)
    
    if not vbench_files:
        print(f"⚠️ VBench에서 타임스탬프 {args.timestamp}와 일치하는 파일을 찾을 수 없습니다.")
    else:
        print(f"✅ 찾은 VBench 결과 파일: {len(vbench_files)}개")
        
        # 4. 파일들을 복사한 폴더에 복사
        for file_path in vbench_files:
            file_name = os.path.basename(file_path)
            target_file = os.path.join(target_folder, file_name)
            shutil.copy2(file_path, target_file)
        
        print(f"✅ VBench 파일 복사 완료: {target_folder}")
    
    # 5. 병합 작업 실행
    print(f"🔄 병합 작업 시작: {target_folder}")
    merge_benchmarks(target_folder)


if __name__ == "__main__":
    main()

