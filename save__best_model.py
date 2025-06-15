from numpy import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --------------------------------------
# 1. Load and preprocess face embeddings
# --------------------------------------
def load_data(filepath='face-embeddings.npz'):
    data = load(filepath)
    X_train, y_train = data['trainX'], data['trainy']
    X_test, y_test = data['testX'], data['testy']

    # Normalize embeddings (L2 norm)
    in_encoder = Normalizer(norm='l2')
    X_train = in_encoder.transform(X_train)
    X_test = in_encoder.transform(X_test)

    # Encode string labels to integers
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train_enc = out_encoder.transform(y_train)
    y_test_enc = out_encoder.transform(y_test)

    return X_train, y_train_enc, X_test, y_test_enc, out_encoder.classes_, in_encoder, out_encoder

# --------------------------------------
# 2. Train and evaluate a given model
# --------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n🔍 [{}] Accuracy: {:.2f}%".format(model.__class__.__name__, acc * 100))
    print("[{}] Classification Report:".format(model.__class__.__name__))
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_pred, acc

# --------------------------------------
# 3. Plot confusion matrices for models
# --------------------------------------
def plot_confusion_matrices(models_preds, y_test, class_names):
    plt.figure(figsize=(18, 5))
    index = 1
    for model_name, y_pred in models_preds.items():
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(1, len(models_preds), index)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(model_name)
        index += 1
    plt.tight_layout()
    plt.show()

# --------------------------------------
# 4. Save model with error handling
# --------------------------------------
def save_model(model, filename):
    try:
        joblib.dump(model, filename, compress=3)
        print("✅ Model saved as '{}'".format(filename))
    except Exception as e:
        print("❌ Failed to save model: {}".format(str(e)))

# --------------------------------------
# 5. Save encoders (for future predictions)
# --------------------------------------
def save_encoders(normalizer, label_encoder):
    try:
        joblib.dump(normalizer, 'normalizer.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
        print("✅ Encoders saved successfully.")
    except Exception as e:
        print("❌ Failed to save encoders: {}".format(str(e)))

# --------------------------------------
# Main function
# --------------------------------------
def main():
    X_train, y_train, X_test, y_test, class_names, in_encoder, out_encoder = load_data()

    # Define models to compare
    models = {
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    }

    models_preds = {}
    accuracies = {}

    # Train and evaluate each model
    for name, model in models.items():
        y_pred, acc = evaluate_model(model, X_train, y_train, X_test, y_test, class_names)
        models_preds[name] = y_pred
        accuracies[name] = acc

    # Determine the best performing model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]
    print("\n🏆 Best Model: {} with Accuracy: {:.2f}%".format(best_model_name, best_accuracy * 100))

    # Plot all confusion matrices
    plot_confusion_matrices(models_preds, y_test, class_names)

    # Save the best model
    model_filename = "{}_model.joblib".format(best_model_name.lower().replace(' ', '_'))
    save_model(best_model, model_filename)

    # Save the encoders
    save_encoders(in_encoder, out_encoder)

    # Show current directory contents
    print("\n📂 Current directory: {}".format(os.getcwd()))
    print("📄 Files: {}".format(os.listdir()))

if __name__ == "__main__":
    main()
