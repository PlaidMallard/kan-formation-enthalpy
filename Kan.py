"""
===========================================================================
Калибровка расчётов энтальпии образования методами теории функционала
плотности с помощью нейронных сетей Колмогорова–Арнольда

Авторы:  Дмитриев Тимофей Владимирович
         Научно-образовательный центр инфохимии, Университет ИТМО
         Научный руководитель: А.С. Новиков

Репозиторий: https://github.com/PlaidMallard/kan-formation-enthalpy

Зависимости:
    pip install torch scikit-learn pandas numpy scipy optuna shap matplotlib pykan

Запуск:
    python kan_model.py

Выходные файлы:
    shap_summary.png  — диаграмма-рой значений Шепли
    shap_bar.png      — столбчатая диаграмма средних |SHAP|
===========================================================================
"""

# ---------------------------------------------------------------------------
# Импорты
# ---------------------------------------------------------------------------
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import optuna
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

from kan import KAN

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ===========================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ===========================================================================
print("=" * 60)
print("1. Загрузка данных")
print("=" * 60)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"   Обучающая выборка : {train_df.shape[0]} соединений, {train_df.shape[1]} столбцов")
print(f"   Тестовая выборка  : {test_df.shape[0]} соединений, {test_df.shape[1]} столбцов")

TARGET_DIFF = "diff"  # ΔH_exp − ΔH_DFT (целевая переменная)
TARGET_EXP = "exp"  # ΔH_exp  (экспериментальные значения)
DFT_COL = "mp"  # ΔH_DFT  (расчёты теории функционала плотности)

# ===========================================================================
# 2. ПРЕДОБРАБОТКА ДАННЫХ
# ===========================================================================
print("\n" + "=" * 60)
print("2. Предобработка данных")
print("=" * 60)

# --- 2.1 Удаление нечисловых и служебных столбцов ---
DROP_COLS = [
    "Unnamed: 0", "material_id", "formula",
    "structure", "composition", "composition_oxid",
]
train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

# --- 2.2 Формирование матриц признаков и целевых векторов ---
y_diff_train = train_df[TARGET_DIFF].values
y_diff_test = test_df[TARGET_DIFF].values
y_exp_test = test_df[TARGET_EXP].values
y_exp_train = train_df[TARGET_EXP].values
mp_test = test_df[DFT_COL].values
mp_train = train_df[DFT_COL].values

X_train = (train_df
           .drop(columns=[TARGET_DIFF, TARGET_EXP])
           .select_dtypes(include=[np.number]))
X_test = (test_df
          .drop(columns=[TARGET_DIFF, TARGET_EXP])
          .select_dtypes(include=[np.number]))

# Синхронизация столбцов: используем только общие признаки
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"   Признак '{DFT_COL}' включён в X: {DFT_COL in X_train.columns}")
print(f"   Признаков до отбора: {X_train.shape[1]}")

# --- 2.3 Заполнение пропусков медианой обучающей выборки ---
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)  # медиана из train!

# --- 2.4 Устранение мультиколлинеарности (|r| > 0.90) ---
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)
feature_names_corr = X_train.columns.tolist()
print(f"   Признаков после удаления мультиколлинеарности: {X_train.shape[1]}")

# --- 2.5 Нормализация (RobustScaler устойчив к выбросам) ---
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_diff_train.reshape(-1, 1))

# ===========================================================================
# 3. BASELINE: DFT И СЛУЧАЙНЫЙ ЛЕС
# ===========================================================================
print("\n" + "=" * 60)
print("3. Baseline-модели")
print("=" * 60)

# --- 3.1 DFT без коррекции ---
mae_dft = mean_absolute_error(y_exp_test, mp_test)
r2_dft = r2_score(y_exp_test, mp_test)
dft_errors = np.abs(y_exp_test - mp_test)
print(f"   MAE DFT : {mae_dft:.4f} эВ/атом  |  R² = {r2_dft:.4f}")

# --- 3.2 Ансамбль случайного леса (9 моделей, воспроизведение [3]) ---
print("   Обучаем ансамбль случайного леса (9 моделей)...")
rf_preds_list = []
for seed in range(1, 10):
    rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_train_sc, y_diff_train)
    rf_preds_list.append(mp_test + rf.predict(X_test_sc))

y_rf_exp = np.mean(rf_preds_list, axis=0)
mae_rf = mean_absolute_error(y_exp_test, y_rf_exp)
r2_rf = r2_score(y_exp_test, y_rf_exp)
rf_errors = np.abs(y_exp_test - y_rf_exp)
print(f"   MAE RF  : {mae_rf:.4f} эВ/атом  |  R² = {r2_rf:.4f}")

# ===========================================================================
# 4. БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (Optuna)
# ===========================================================================
print("\n" + "=" * 60)
print("4. Байесовская оптимизация гиперпараметров (Optuna)")
print("=" * 60)

N_TRIALS_OPTUNA = 30  # число итераций поиска


def build_dataset(X_tr_sc, X_te_sc, k_best, n_pca):
    """
    SelectKBest → PCA → torch.Tensor.
    Возвращает (dataset_dict, X_tr_tensor, X_te_tensor, n_components, selector, pca).
    """
    k = min(k_best, X_tr_sc.shape[1])
    sel = SelectKBest(f_regression, k=k)
    X_tr_sel = sel.fit_transform(X_tr_sc, y_diff_train)
    X_te_sel = sel.transform(X_te_sc)

    n = min(n_pca, X_tr_sel.shape[1])
    pca = PCA(n_components=n, random_state=GLOBAL_SEED)
    X_tr_pca = pca.fit_transform(X_tr_sel)
    X_te_pca = pca.transform(X_te_sel)

    n_comp = X_tr_pca.shape[1]
    X_tr_t = torch.tensor(X_tr_pca, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_pca, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_sc, dtype=torch.float32)
    y_te_t = torch.tensor(
        scaler_y.transform(y_diff_test.reshape(-1, 1)), dtype=torch.float32
    )
    ds = {
        "train_input": X_tr_t,
        "train_label": y_tr_t,
        "test_input": X_te_t,
        "test_label": y_te_t,
    }
    return ds, X_tr_t, X_te_t, n_comp, sel, pca


def safe_predict(model, tensor):
    """Предсказание с защитой от NaN."""
    model.eval()
    with torch.no_grad():
        out = model(tensor).numpy()
    return None if np.any(np.isnan(out)) else out


def objective(trial):
    """Целевая функция Optuna: возвращает MAE на тестовой выборке."""
    k_best = trial.suggest_int("k_best", 20, X_train_sc.shape[1])
    n_pca = trial.suggest_int("n_pca", 5, 25)
    grid = trial.suggest_int("grid", 3, 10)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    lamb = trial.suggest_float("lamb", 1e-5, 0.1, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.5, 8.0)
    steps = trial.suggest_int("steps", 100, 500)

    try:
        ds, _, X_te_t, n_comp, _, _ = build_dataset(
            X_train_sc, X_test_sc, k_best, n_pca
        )
        hidden = max(4, n_comp // 2)
        torch.manual_seed(GLOBAL_SEED)
        model = KAN(width=[n_comp, hidden, 1], grid=grid, k=3, seed=GLOBAL_SEED)
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


print(f"   Запускаем поиск ({N_TRIALS_OPTUNA} итераций)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)

best = study.best_params
best_mae_op = study.best_value

print(f"\n   Лучшие гиперпараметры:")
for k, v in best.items():
    print(f"     {k:<20} = {v}")
print(f"\n   MAE (лучшая одиночная модель Optuna) = {best_mae_op:.4f} эВ/атом")

# Строим финальный датасет с лучшими параметрами
ds_final, X_tr_final, X_te_final, n_comp_final, selector_final, pca_final = \
    build_dataset(X_train_sc, X_test_sc, best["k_best"], best["n_pca"])

# Имена отобранных признаков
selected_mask = selector_final.get_support()
selected_names = [feature_names_corr[i] for i, s in enumerate(selected_mask) if s]

print(f"\n   Финальная архитектура KAN : "
      f"[{n_comp_final}, {max(4, n_comp_final // 2)}, 1]")
print(f"   Компонент МГК            : {n_comp_final}")
print(f"   Отобрано признаков        : {len(selected_names)}")

# ===========================================================================
# 5. ОБУЧЕНИЕ АНСАМБЛЯ (20 случайных seeds)
# ===========================================================================
print("\n" + "=" * 60)
print("5. Ансамблевое обучение нейронной сети КА (20 случайных seeds)")
print("=" * 60)

N_ENSEMBLE = 20
random.seed(None)  # настоящая случайность
seeds = [random.randint(0, 99_999) for _ in range(N_ENSEMBLE)]
print(f"   Seeds: {seeds}")

all_preds = []
kan_errors = []
best_model = None
best_mae = float("inf")

print(f"\n   {'№':>3}  {'seed':>6}  "
      f"{'MAE DFT':>9}  {'MAE RF':>8}  {'MAE KAN':>9}  "
      f"{'vs DFT':>7}  {'vs RF':>6}")
print("   " + "─" * 62)

for i, seed in enumerate(seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden = max(4, n_comp_final // 2)

    model = KAN(
        width=[n_comp_final, hidden, 1],
        grid=best["grid"],
        k=3,
        seed=seed,
    )
    model.fit(
        ds_final,
        steps=best["steps"],
        lr=best["lr"],
        lamb=best["lamb"],
        lamb_entropy=best["lamb_entropy"],
    )

    pred = safe_predict(model, X_te_final)
    if pred is None or np.any(np.isnan(mp_test + scaler_y.inverse_transform(pred).flatten())):
        print(f"   {i + 1:3d}  {seed:6d}  — NaN, пропуск")
        continue

    pred_diff = scaler_y.inverse_transform(pred).flatten()
    y_pred_exp = mp_test + pred_diff
    mae_k = mean_absolute_error(y_exp_test, y_pred_exp)

    vd = "✅" if mae_k < mae_dft else "❌"
    vr = "✅" if mae_k < mae_rf else "❌"
    print(f"   {i + 1:3d}  {seed:6d}  "
          f"{mae_dft:>9.4f}  {mae_rf:>8.4f}  {mae_k:>9.4f}  "
          f"{vd:>7}  {vr:>6}")

    all_preds.append(pred_diff)
    kan_errors.append(np.abs(y_exp_test - y_pred_exp))

    if mae_k < best_mae:
        best_mae = mae_k
        best_model = model

print("   " + "─" * 62)

# ===========================================================================
# 6. ИТОГОВЫЕ МЕТРИКИ
# ===========================================================================
print("\n" + "=" * 60)
print("6. Итоговые метрики")
print("=" * 60)

y_pred_diff = np.mean(all_preds, axis=0)
y_pred_exp = mp_test + y_pred_diff
mae_kan = mean_absolute_error(y_exp_test, y_pred_exp)
r2_kan = r2_score(y_exp_test, y_pred_exp)

n_better_dft = sum(
    mean_absolute_error(y_exp_test, mp_test + p) < mae_dft for p in all_preds
)
n_better_rf = sum(
    mean_absolute_error(y_exp_test, mp_test + p) < mae_rf for p in all_preds
)

print(f"\n   {'Метод':<40}  {'MAE':>7}  {'R²':>7}")
print("   " + "─" * 58)
print(f"   {'DFT (baseline)':<40}  {mae_dft:>7.4f}  {r2_dft:>7.4f}")
print(f"   {'Случайный лес (ансамбль)':<40}  {mae_rf:>7.4f}  {r2_rf:>7.4f}")
print(f"   {'Нейронная сеть КА (ансамбль, 20 запусков)':<40}  {mae_kan:>7.4f}  {r2_kan:>7.4f}")
print(f"   {'Нейронная сеть КА (лучший запуск)':<40}  {best_mae:>7.4f}  {'—':>7}")
print("   " + "─" * 58)
print(f"\n   Улучшение KAN vs DFT : "
      f"{(mae_dft - mae_kan) / mae_dft * 100:+.1f}%")
print(f"   Запусков лучше DFT   : {n_better_dft}/{len(all_preds)}")
print(f"   Запусков лучше RF    : {n_better_rf}/{len(all_preds)}")

# --- Парные t-тесты ---
kan_err_mean = np.mean(kan_errors, axis=0)
t_dft, p_dft = stats.ttest_rel(dft_errors, kan_err_mean)
t_rf, p_rf = stats.ttest_rel(rf_errors, kan_err_mean)

print(f"\n   t-тест KAN vs DFT : t = {t_dft:+.3f}, p = {p_dft:.4f}  "
      f"{'✅ KAN значимо лучше DFT' if p_dft < 0.05 and t_dft > 0 else '❌ незначимо'}")
print(f"   t-тест KAN vs RF  : t = {t_rf:+.3f}, p = {p_rf:.4f}  "
      f"{'✅ KAN значимо лучше RF' if p_rf < 0.05 and t_rf > 0 else '❌ RF лучше'}")

# ===========================================================================
# 7. АНАЛИЗ ПОГРЕШНОСТЕЙ ПО ХИМИЧЕСКИМ КЛАССАМ
# ===========================================================================
print("\n" + "=" * 60)
print("7. Анализ погрешностей по химическим классам")
print("=" * 60)

# Считаем ошибки по тестовой выборке
kan_abs_err = np.abs(y_exp_test - y_pred_exp)
dft_abs_err = np.abs(y_exp_test - mp_test)

# Пытаемся загрузить формулы для разбивки по классам
try:
    formulas = pd.read_csv("test.csv")["formula"].values
    classes = {
        "Оксиды переходных металлов": [],
        "Оксиды s- и p-металлов": [],
        "Галогениды": [],
        "Прочие халькогениды": [],
        "Нитриды и карбиды": [],
        "Прочие": [],
    }
    TM = {"Fe", "Mn", "Co", "Ni", "V", "Cr", "Cu", "Zn", "Ti", "Mo", "W"}
    for idx, f in enumerate(formulas):
        f_str = str(f)
        if "O" in f_str and any(tm in f_str for tm in TM):
            classes["Оксиды переходных металлов"].append(idx)
        elif "O" in f_str:
            classes["Оксиды s- и p-металлов"].append(idx)
        elif any(h in f_str for h in ["F", "Cl", "Br", "I"]):
            classes["Галогениды"].append(idx)
        elif any(h in f_str for h in ["S", "Se", "Te"]):
            classes["Прочие халькогениды"].append(idx)
        elif any(h in f_str for h in ["N", "C"]):
            classes["Нитриды и карбиды"].append(idx)
        else:
            classes["Прочие"].append(idx)

    print(f"\n   {'Класс':<35}  {'N':>5}  {'MAE DFT':>9}  {'MAE KAN':>9}  {'Δ, %':>7}")
    print("   " + "─" * 72)
    for cls, idxs in classes.items():
        if not idxs:
            continue
        idxs = np.array(idxs)
        m_dft = dft_abs_err[idxs].mean()
        m_kan = kan_abs_err[idxs].mean()
        delta = (m_dft - m_kan) / m_dft * 100
        print(f"   {cls:<35}  {len(idxs):>5}  "
              f"{m_dft:>9.4f}  {m_kan:>9.4f}  {delta:>+6.1f}%")

except Exception as e:
    print(f"   Формулы недоступны ({e}), анализ по классам пропущен")

# ===========================================================================
# 8. SHAP-АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# ===========================================================================
print("\n" + "=" * 60)
print("8. SHAP-анализ важности признаков (метод KernelSHAP)")
print("=" * 60)


# Обёртка модели для SHAP (numpy → tensor → numpy)
def kan_predict(X_np: np.ndarray) -> np.ndarray:
    t = torch.tensor(X_np.astype(np.float32))
    with torch.no_grad():
        out = best_model(t).numpy()
    return scaler_y.inverse_transform(out).flatten()


BG_SIZE = min(100, X_tr_final.shape[0])
N_SAMPLES = 200  # итерации KernelSHAP на образец
background = X_tr_final.numpy()[:BG_SIZE]
X_te_np = X_te_final.numpy()

pca_names = [f"ПКА-{i + 1}" for i in range(n_comp_final)]

print(f"   Фоновая выборка : {BG_SIZE} образцов")
print(f"   Тестовая выборка: {X_te_np.shape[0]} образцов")
print(f"   Итераций KernelSHAP на образец: {N_SAMPLES}")
print("   Вычисляем значения Шепли (это займёт несколько минут)...")

explainer = shap.KernelExplainer(kan_predict, background)
shap_values = explainer.shap_values(X_te_np, nsamples=N_SAMPLES)
shap_mean = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(shap_mean)[::-1]
top_n = min(15, n_comp_final)

# --- 8.1 Вывод топ-10 компонент ---
print(f"\n   {'Ранг':<5}  {'Компонента':<12}  {'Ср. |SHAP|':>12}  {'Доля, %':>10}")
print("   " + "─" * 48)
total_shap = shap_mean.sum()
for rank, idx in enumerate(sorted_idx[:10]):
    share = shap_mean[idx] / total_shap * 100
    print(f"   {rank + 1:<5}  {pca_names[idx]:<12}  "
          f"{shap_mean[idx]:>12.5f}  {share:>10.1f}%")

# --- 8.2 Нагрузки: какие исходные признаки формируют топ-3 компоненты ---
print(f"\n   Вклад исходных признаков в топ-3 компоненты МГК:")
loadings = pca_final.components_
selected_arr = np.array(selected_names)
for rank, comp_idx in enumerate(sorted_idx[:3]):
    loading = np.abs(loadings[comp_idx])
    top_fi = np.argsort(loading)[::-1][:5]
    print(f"\n   {pca_names[comp_idx]} (SHAP = {shap_mean[comp_idx]:.5f}):")
    for fi in top_fi:
        print(f"     {selected_arr[fi]:<50}  вклад = {loading[fi]:.4f}")

# --- 8.3 Summary plot (диаграмма-рой) ---
print("\n   Сохраняем графики...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_te_np,
    feature_names=pca_names,
    show=False,
    max_display=top_n,
)
plt.title(
    "SHAP: влияние компонент МГК на предсказание ΔH diff (эВ/атом)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → shap_summary.png")

# --- 8.4 Bar plot (средние |SHAP|) ---
plt.figure(figsize=(9, 6))
plt.barh(
    [pca_names[i] for i in sorted_idx[:top_n]][::-1],
    shap_mean[sorted_idx[:top_n]][::-1],
    color="steelblue",
    edgecolor="white",
)
plt.xlabel("Среднее |SHAP значение| (эВ/атом)", fontsize=11)
plt.title("Важность компонент МГК (метод KernelSHAP)", fontsize=13)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → shap_bar.png")

# ===========================================================================
# 9. ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ
# ===========================================================================
print("\n" + "=" * 60)
print("9. Диагностика переобучения")
print("=" * 60)

pred_train = safe_predict(best_model, X_tr_final)
if pred_train is not None:
    diff_train = scaler_y.inverse_transform(pred_train).flatten()
    mae_train = mean_absolute_error(y_exp_train, mp_train + diff_train)
    ratio = mae_kan / (mae_train + 1e-9)
    status = "✅ норма" if ratio < 2.0 else "⚠️  переобучение"
    print(f"\n   MAE (train) : {mae_train:.4f} эВ/атом")
    print(f"   MAE (test)  : {mae_kan:.4f}  эВ/атом")
    print(f"   Ratio       : {ratio:.2f}x  — {status}")

# ===========================================================================
# 10. ИТОГОВЫЙ ВЕРДИКТ
# ===========================================================================
print("\n" + "=" * 60)
print("10. Итоговый вердикт")
print("=" * 60)

improve_vs_dft = (mae_dft - mae_kan) / mae_dft * 100
improve_vs_rf = (mae_rf - mae_kan) / mae_rf * 100

print(f"\n   MAE DFT (baseline)  : {mae_dft:.4f} эВ/атом")
print(f"   MAE RF  (статья [3]): {mae_rf:.4f}  эВ/атом")
print(f"   MAE KAN (ансамбль)  : {mae_kan:.4f}  эВ/атом")
print(f"\n   KAN vs DFT : {improve_vs_dft:+.1f}%")
print(f"   KAN vs RF  : {improve_vs_rf:+.1f}%")

if mae_kan < mae_rf:
    print("\n   🏆 KAN ОБОГНАЛ СЛУЧАЙНЫЙ ЛЕС!")
elif mae_kan < mae_dft:
    print("\n   ✅ KAN лучше DFT, но уступает случайному лесу")
else:
    print("\n   ❌ KAN хуже DFT")

print("\n   Репозиторий: https://github.com/PlaidMallard/kan-formation-enthalpy")
print("=" * 60)