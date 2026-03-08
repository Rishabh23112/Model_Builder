import sys
from dataset_analyzer import analyze_dataset
from model_generator import build_model
from trainer import train_model
from preprocess import preprocess_data
from evaluator import evaluate_model


if len(sys.argv) != 2:
    print("Usage: python main.py datasets/<csv_file>")
    print("No CSV file found. Using default: datasets/ai_jobs_market_2025_2026.csv")
    csv_file = "datasets/ai_jobs_market_2025_2026.csv"
else:
    csv_file = sys.argv[1]
df, info = analyze_dataset(csv_file)

print(info)

train_loader, x_test, y_test, num_features = preprocess_data(df)

model = build_model(num_features, info['num_classes'])

train_model(model, train_loader, info['task'], epochs=5)
evaluate_model(model, x_test, y_test)
