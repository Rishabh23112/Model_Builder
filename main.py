import sys
from dataset_analyzer import analyze_dataset
from model_generator import build_model
from trainer import train_model
from preprocess import preprocess_data
from evaluator import evaluate_model
from logs import log_details
from model_selector import select_model


if len(sys.argv) != 2:
    print("Usage: python main.py datasets/<csv_file>")
    print("No CSV file found. Using default: datasets/ai_jobs_market_2025_2026.csv")
    csv_file = "datasets/ai_jobs_market_2025_2026.csv"
else:
    csv_file = sys.argv[1]
df, info = analyze_dataset(csv_file)

print(info)

info['dataset_name'] = csv_file

train_loader, x_test, y_test, num_features = preprocess_data(df, info['task'])

if info["task"] == "classification":
    output_size = info["num_classes"]
else:
    output_size = 1

model = build_model(num_features, output_size)

train_model(model, train_loader, info['task'], epochs=5)
accuracy=evaluate_model(model, x_test, y_test, info['task'])

x_train_np = train_loader.dataset.tensors[0].numpy()
y_train_np = train_loader.dataset.tensors[1].numpy().ravel()

x_test_np = x_test.numpy()
y_test_np = y_test.numpy().ravel()

results, best_model = select_model(
    x_train_np,
    y_train_np,
    x_test_np,
    y_test_np,
    info['task']
)

print("\nSklearn Model Results")
print("----------------------")

for model_name, score in results.items():
    print(model_name, ":", score)

print("\nBest Model:", best_model)


log_details(info, accuracy=accuracy, best_model=best_model)
