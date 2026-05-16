import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("train.csv")

# Убираем пробелы в названиях
df.columns = df.columns.str.strip()

print("Колонки в датасете:")
print(df.columns)

# =========================
# 2. УКАЖИ ЗДЕСЬ НАЗВАНИЯ
# =========================
# ❗ ВАЖНО: замени под свой датасет после print(df.columns)

EXP_COL = "exp"   # ← замени!
DFT_COL = "mp"    # ← замени!

# =========================
# 3. Проверка колонок
# =========================
if EXP_COL not in df.columns or DFT_COL not in df.columns:
    raise ValueError("❌ Проверь названия колонок EXP_COL и DFT_COL")

# =========================
# 4. Target (multifidelity)
# =========================
y_exp = df[EXP_COL]
y_dft = df[DFT_COL]

# ΔHfdiff
y_diff = y_exp - y_dft

# =========================
# 5. Признаки
# =========================

X = df.select_dtypes(include=[np.number]).copy()

# УБИРАЕМ УТЕЧКУ
X = X.drop(columns=["exp", "mp", "diff"], errors="ignore")

# =========================
# 6. Обработка NaN
# =========================
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# =========================
# 7. Train/Test split
# =========================
X_train, X_test, y_train_diff, y_test_diff, y_train_dft, y_test_dft, y_train_exp, y_test_exp = train_test_split(
    X, y_diff, y_dft, y_exp,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =========================
# 8. Модель RF
# =========================
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("Обучение модели...")
model.fit(X_train, y_train_diff)

# =========================
# 9. Предсказание
# =========================
y_pred_diff = model.predict(X_test)

# Восстанавливаем exp
y_pred_exp = y_test_dft + y_pred_diff

# =========================
# 10. Метрики
# =========================
mae_ml = mean_absolute_error(y_test_exp, y_pred_exp)
r2_ml = r2_score(y_test_exp, y_pred_exp)

# baseline DFT
mae_dft = mean_absolute_error(y_test_exp, y_test_dft)

print("\n📊 РЕЗУЛЬТАТЫ:")
print(f"MAE DFT : {mae_dft:.4f}")
print(f"MAE ML  : {mae_ml:.4f}")
print(f"R2 ML   : {r2_ml:.4f}")

# =========================
# 11. Проверка успеха
# =========================
if mae_ml < mae_dft:
    print("✅ ML лучше DFT (всё работает как в статье)")
else:
    print("⚠️ ML хуже DFT — надо улучшать модель")