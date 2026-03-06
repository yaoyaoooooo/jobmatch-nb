from jobmatch_nb.paths import RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR

for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("目录创建完成。")