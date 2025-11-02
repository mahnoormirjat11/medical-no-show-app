import joblib

# Save trained model and label encoder
joblib.dump(model, "no_show_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model and encoder saved successfully!")
