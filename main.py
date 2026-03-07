from dataset_analyzer import analyze_dataset
from model_generator import build_model

df, info=analyze_dataset('ai_jobs_market_2025_2026.csv')

print(info)

model=build_model(4,3)
print(model)