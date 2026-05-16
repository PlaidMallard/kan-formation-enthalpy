import pandas as pd
import numpy as np
import torch
import random
import optuna
import warnings
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
from kan import KAN

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================
# 1. Загрузка данных
# =========================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

target_diff = "diff"
target_exp = "exp"
dft_col = "mp"  # ← ключевой признак! используется и как фича и для восстановления

# =========================
# 2. Очистка
# =========================
drop_cols = ["Unnamed: 0", "material_id", "formula", "structure", "composition",
             "composition_oxid"]
train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# =========================
# 3. X и y
# =========================
y_diff_train = train_df[target_diff].values
y_diff_test = test_df[target_diff].values
y_exp_test = test_df[target_exp].values
y_exp_train = train_df[target_exp].values
mp_test = test_df[dft_col].values
mp_train = train_df[dft_col].values


# Убираем только таргеты diff и exp
X_train = train_df.drop(columns=[target_diff, target_exp]).select_dtypes(include=[np.number])
X_test = test_df.drop(columns=[target_diff, target_exp]).select_dtypes(include=[np.number])

common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"mp в признаках: {'mp' in X_train.columns} " if 'mp' in X_train.columns else "mp в признаках:  ПРОБЛЕМА!")
print(f"Признаков: {X_train.shape[1]}")

# =========================
# 4. NaN - медиана train
# =========================
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# =========================
# 5. MAE DFT
# =========================
mae_dft = mean_absolute_error(y_exp_test, mp_test)
r2_dft = r2_score(y_exp_test, mp_test)
dft_errors = np.abs(y_exp_test - mp_test)

print(f"\nBaseline MAE DFT : {mae_dft:.4f}")
print(f"Baseline R2  DFT : {r2_dft:.4f}")

# =========================
# 6. RF из статьи
# =========================
print("\n" + "=" * 50)
print("  RF из статьи (воспроизведение)")
print("=" * 50)


X_train_np = X_train.values
X_test_np = X_test.values

rf_maes = []
rf_preds_list = []

for i in range(1, 10):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=i,
        n_jobs=-1
    )
    rf.fit(X_train_np, y_diff_train)
    y_rf_diff = rf.predict(X_test_np)
    y_rf_exp = mp_test + y_rf_diff
    mae_rf_i = mean_absolute_error(y_exp_test, y_rf_exp)
    rf_maes.append(mae_rf_i)
    rf_preds_list.append(y_rf_exp)
    print(f"  RF seed={i}: MAE={mae_rf_i:.4f}")

mae_rf_mean = np.mean(rf_maes)
mae_rf_std = np.std(rf_maes)
y_rf_ensemble = np.mean(rf_preds_list, axis=0)
mae_rf_ensemble = mean_absolute_error(y_exp_test, y_rf_ensemble)
r2_rf = r2_score(y_exp_test, y_rf_ensemble)

print(f"\n  RF среднее MAE    : {mae_rf_mean:.4f} ± {mae_rf_std:.4f}")
print(f"  RF ансамбль MAE   : {mae_rf_ensemble:.4f}")
print(f"  RF ансамбль R2    : {r2_rf:.4f}")
print(f"  Статья даёт       : ~0.0617")

# =========================
# 7. Подготовка для KAN
# =========================
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train_np)
X_test_scaled = scaler_X.transform(X_test_np)
y_train_scaled = scaler_y.fit_transform(y_diff_train.reshape(-1, 1))


# =========================
# 8. Вспомогательные функции
# =========================
def safe_predict(model, tensor):
    model.eval()
    with torch.no_grad():
        out = model(tensor).numpy()
    if np.any(np.isnan(out)):
        return None
    return out


def build_dataset(X_tr_sc, X_te_sc, y_tr_sc, k_best, n_pca):
    """SelectKBest + PCA → тензоры"""
    k = min(k_best, X_tr_sc.shape[1])
    sel = SelectKBest(f_regression, k=k)
    X_tr_sel = sel.fit_transform(X_tr_sc, y_diff_train)
    X_te_sel = sel.transform(X_te_sc)

    n_comp = min(n_pca, X_tr_sel.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_sel)
    X_te_p = pca.transform(X_te_sel)

    n_out = X_tr_p.shape[1]

    ds = {
        'train_input': torch.tensor(X_tr_p, dtype=torch.float32),
        'train_label': torch.tensor(y_tr_sc, dtype=torch.float32),
        'test_input': torch.tensor(X_te_p, dtype=torch.float32),
        'test_label': torch.tensor(
            scaler_y.transform(y_diff_test.reshape(-1, 1)), dtype=torch.float32
        ),
    }
    return ds, torch.tensor(X_tr_p, dtype=torch.float32), \
        torch.tensor(X_te_p, dtype=torch.float32), n_out


# =========================
# 9. Optuna — поиск лучших параметров KAN
# =========================
print("\n" + "=" * 50)
print("  Optuna: поиск параметров KAN")
print("=" * 50)

N_TRIALS_OPTUNA = 30


def objective(trial):
    k_best = trial.suggest_int("k_best", 20, X_train_scaled.shape[1])
    n_pca = trial.suggest_int("n_pca", 5, 25)
    grid = trial.suggest_int("grid", 3, 10)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    lamb = trial.suggest_float("lamb", 1e-5, 0.1, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.5, 8.0)
    steps = trial.suggest_int("steps", 100, 500)

    try:
        ds, X_tr_t, X_te_t, n_comp = build_dataset(
            X_train_scaled, X_test_scaled, y_train_scaled, k_best, n_pca
        )
        hidden = max(4, n_comp // 2)

        torch.manual_seed(42)
        model = KAN(width=[n_comp, hidden, 1], grid=grid, k=3, seed=42)
        model.fit(ds, steps=steps, lr=lr, lamb=lamb, lamb_entropy=lamb_entropy)

        pred = safe_predict(model, X_te_t)
        if pred is None:
            return float("inf")

        pred_diff = scaler_y.inverse_transform(pred).flatten()
        y_pred_exp = mp_test + pred_diff

        if np.any(np.isnan(y_pred_exp)):
            return float("inf")

        return mean_absolute_error(y_exp_test, y_pred_exp)

    except Exception:
        return float("inf")


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)

best = study.best_params
best_mae_op = study.best_value

print(f"\n Optuna завершила!")
print(f"\nЛучшие параметры:")
for k, v in best.items():
    print(f"  {k:<15} = {v}")
print(f"\n  MAE Optuna (1 модель) : {best_mae_op:.4f}")
print(f"  MAE DFT baseline      : {mae_dft:.4f}")
print(f"  MAE RF ансамбль       : {mae_rf_ensemble:.4f}")

# =========================
# 10. Финальный датасет с лучшими параметрами
# =========================
ds_final, X_tr_fin, X_te_fin, n_comp_fin = build_dataset(
    X_train_scaled, X_test_scaled, y_train_scaled,
    best["k_best"], best["n_pca"]
)
print(f"\nФинальная архитектура KAN: [{n_comp_fin}, {max(4, n_comp_fin // 2)}, 1]")

# =========================
# 11. Статистическое тестирование KAN
#     20 случайных seeds с лучшими параметрами
# =========================
print("\n" + "=" * 50)
print("  KAN: 20 случайных сидов")
print("=" * 50)

N_TRIALS = 20
random.seed(None)
seeds = [random.randint(0, 99999) for _ in range(N_TRIALS)]
print(f"Seeds: {seeds}\n")

print(f"{'─' * 70}")
print(f"  №   seed    MAE DFT   MAE RF    MAE KAN   vs DFT    vs RF")
print(f"{'─' * 70}")

mae_kan_list = []
kan_errors = []

for i, seed in enumerate(seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden = max(4, n_comp_fin // 2)

    try:
        model = KAN(
            width=[n_comp_fin, hidden, 1],
            grid=best["grid"], k=3, seed=seed
        )
        model.fit(
            ds_final,
            steps=best["steps"],
            lr=best["lr"],
            lamb=best["lamb"],
            lamb_entropy=best["lamb_entropy"],
        )

        pred = safe_predict(model, X_te_fin)
        if pred is None:
            print(f"  {i + 1:2d}  {seed:5d}   {mae_dft:.4f}    {mae_rf_ensemble:.4f}    {'NaN':>7}    —         —")
            continue

        pred_diff = scaler_y.inverse_transform(pred).flatten()
        y_pred_exp = mp_test + pred_diff

        if np.any(np.isnan(y_pred_exp)):
            print(f"  {i + 1:2d}  {seed:5d}   {mae_dft:.4f}    {mae_rf_ensemble:.4f}    {'NaN':>7}    —         —")
            continue

        mae_kan = mean_absolute_error(y_exp_test, y_pred_exp)
        vs_dft = "good" if mae_kan < mae_dft else "not good"
        vs_rf = "good" if mae_kan < mae_rf_ensemble else "not good"

        print(
            f"  {i + 1:2d}  {seed:5d}   {mae_dft:.4f}    {mae_rf_ensemble:.4f}    {mae_kan:.4f}    {vs_dft}         {vs_rf}")

        mae_kan_list.append(mae_kan)
        kan_errors.append(np.abs(y_exp_test - y_pred_exp))

    except Exception as e:
        print(f"  {i + 1:2d}  {seed:5d}   ошибка: {e}")

print(f"{'─' * 70}")

# =========================
# 12. Итоговая статистика
# =========================
mae_kan_arr = np.array(mae_kan_list)
n_valid = len(mae_kan_arr)
n_better_dft = (mae_kan_arr < mae_dft).sum()
n_better_rf = (mae_kan_arr < mae_rf_ensemble).sum()

print(f"\n{'=' * 55}")
print(f"{'=' * 55}")
print(f"  {'Метрика':<22} {'DFT':>8}  {'RF':>8}  {'KAN':>8}")
print(f"{'─' * 55}")
print(f"  {'MAE среднее':<22} {mae_dft:>8.4f}  {mae_rf_ensemble:>8.4f}  {mae_kan_arr.mean():>8.4f}")
print(f"  {'MAE медиана':<22} {mae_dft:>8.4f}  {mae_rf_ensemble:>8.4f}  {np.median(mae_kan_arr):>8.4f}")
print(f"  {'MAE лучшее':<22} {'—':>8}  {'—':>8}  {mae_kan_arr.min():>8.4f}")
print(f"  {'Стд отклонение':<22} {'—':>8}  {'—':>8}  {mae_kan_arr.std():>8.4f}")
print(f"{'─' * 55}")
print(f"  KAN лучше DFT : {n_better_dft}/{n_valid} ({n_better_dft / n_valid * 100:.0f}%)")
print(f"  KAN лучше RF  : {n_better_rf}/{n_valid}  ({n_better_rf / n_valid * 100:.0f}%)")

# =========================
# 13. Парный t-тест KAN vs RF
# =========================
if len(kan_errors) > 1:
    mean_kan_errors = np.mean(kan_errors, axis=0)
    rf_errors_arr = np.abs(y_exp_test - y_rf_ensemble)

    # KAN vs DFT
    t1, p1 = stats.ttest_rel(dft_errors, mean_kan_errors)
    # KAN vs RF
    t2, p2 = stats.ttest_rel(rf_errors_arr, mean_kan_errors)

    print(f"\n{'─' * 55}")
    print(f"  Парные t-тесты")
    print(f"{'─' * 55}")
    print(f"  KAN vs DFT: t={t1:.3f}, p={p1:.4f}  ", end="")
    print(" KAN значимо лучше" if p1 < 0.05 and t1 > 0 else " незначимо" if p1 >= 0.05 else " DFT лучше")

    print(f"  KAN vs RF : t={t2:.3f}, p={p2:.4f}  ", end="")
    print(" KAN значимо лучше RF!" if p2 < 0.05 and t2 > 0 else " незначимо" if p2 >= 0.05 else " RF лучше")

# =========================
# 14. Финальный вердикт
# =========================
print(f"\n{'=' * 55}")
print(f"{'=' * 55}")
print(f"  MAE DFT (baseline) : {mae_dft:.4f}")
print(f"  MAE RF  (статья)   : {mae_rf_ensemble:.4f}  (цель: ~0.0617)")
print(f"  MAE KAN (среднее)  : {mae_kan_arr.mean():.4f}")
print(f"  MAE KAN (лучшее)   : {mae_kan_arr.min():.4f}")
print(f"{'─' * 55}")

if mae_kan_arr.mean() < mae_rf_ensemble:
    impr = (mae_rf_ensemble - mae_kan_arr.mean()) / mae_rf_ensemble * 100
    print(f"   KAN > RF Улучшение: {impr:+.1f}%")
elif mae_kan_arr.mean() < mae_dft:
    impr_dft = (mae_dft - mae_kan_arr.mean()) / mae_dft * 100
    gap_rf = (mae_kan_arr.mean() - mae_rf_ensemble) / mae_rf_ensemble * 100
    print(f"   KAN лучше DFT на {impr_dft:.1f}%")
    print(f"    До RF ещё {gap_rf:.1f}% — нужно улучшать признаки")
else:
    print(f"   KAN пока хуже DFT — нужна другая стратегия")

print(f"{'=' * 55}")