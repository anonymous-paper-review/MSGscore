#!/bin/bash

EC_path="/app/host_volume/MSGscore_git/MSGscore" # put your project files path here
base_videos_dir="/app/host_volume/inference_videos/outputs_Alien_in_Redplanet" # put your video files path here
prompt_dir="/app/host_volume/inference_videos/prompt.csv" # put your prompt file path here
# prompt should be written as "file_path:prompt" in one line
results_dir="result" # put your result output path here
timestamp=$(date +%Y%m%d_%H%M%S)

export CUDA_VISIBLE_DEVICES=0

cd "$EC_path"/utils
python generate_prompt.py --ec_path "$EC_path" --prompt_dir "$prompt_dir" --output_dir "$base_videos_dir" --timestamp "$timestamp"

# results folder creation
if [ ! -d "$EC_path/$results_dir" ]; then
    echo "⚙️  results folder created : $results_dir"
    mkdir -p "$EC_path/$results_dir"
fi

# EvalCrafter

conda init
conda activate EvalCrafter

EvalCrafter_path="$EC_path/EvalCrafter"

find "$base_videos_dir" -type d -links 2 | while read -r dir_videos; do

    echo "=========="
    echo "Evaluation start for: $dir_videos"
    echo "=========="

    folder_name=$(basename "$dir_videos")

    cd "$EvalCrafter_path/metrics"

    echo "[IS]"
    python3 is.py --dir_videos "$dir_videos" --result_dir "$results_dir"

    echo "[VQA_A and VQA_T]"
    cd "$EvalCrafter_path/metrics/DOVER"
    python3 evaluate_a_set_of_videos.py --dir_videos "$dir_videos" --result_dir "$results_dir"

    echo "[CLIP-Score]"
    cd "$EvalCrafter_path/metrics/Scores_with_CLIP"
    python3 Scores_with_CLIP.py --dir_videos "$dir_videos" --metric 'clip_score' --result_dir "$results_dir"

    echo "[Face Consistency]"
    python3 Scores_with_CLIP.py --dir_videos "$dir_videos" --metric 'face_consistency_score' --result_dir "$results_dir"

    echo "[SD-Score]"
    python3 Scores_with_CLIP.py --dir_videos "$dir_videos" --metric 'sd_score' --result_dir "$results_dir"

    echo "[BLIP-BLUE]"
    python3 Scores_with_CLIP.py --dir_videos "$dir_videos" --metric 'blip_bleu' --result_dir "$results_dir"

    echo "[CLIP-Temp]"
    python3 Scores_with_CLIP.py --dir_videos "$dir_videos" --metric 'clip_temp_score' --result_dir "$results_dir"

    echo "[Flow-Score]"
    cd "$EvalCrafter_path/metrics/RAFT"
    python3 optical_flow_scores.py --dir_videos "$dir_videos" --metric 'flow_score' --result_dir "$results_dir"

    echo "[Warping Error]"
    python3 optical_flow_scores.py --dir_videos "$dir_videos" --metric 'warping_error' --result_dir "$results_dir"

    if [ -d "$EC_path/$results_dir" ]; then
        new_results="${EvalCrafter_path}/${results_dir}_${timestamp}_${folder_name}"
        mkdir -p "$(dirname "$new_results")"
        mv "$EC_path/$results_dir" "$new_results"
        echo "📦 Result (EvalCrafter) : $results_dir → $new_results"
    fi

done


# VBench
VBench_path="$EC_path/VBench"
conda activate vbench

cd "$VBench_path
DIMENSION=('subject_consistency' 'background_consistency' 'temporal_flickering' 'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' 'temporal_style' 'overall_consistency')

find "$base_videos_dir" -type d -links 2 | while read -r folder; do
    folder_name=$(basename "$folder")
    echo "🌀 Processing: $folder"

    /opt/conda/bin/python evaluate.py \
        --dimension "${DIMENSION[@]}" \
        --videos_path "$folder" \
        --prompt_file ./prompts/prompts.json \
        --mode=custom_input \
        --timestamp "$timestamp"
done



# MSG score

cd "$EC_path"/utils
python merge_results.py --ec_path "$EC_path" --prompt_dir "$prompt_dir" --output_dir "$base_videos_dir" --timestamp "$timestamp"

cd "$EC_path"
python inference.py --video_dir "$base_videos_dir" --output_dir "$results_dir" --prompt_file "./result/result_$timestamp/merged_results.json" --timestamp "$timestamp"


