from pymatgen.ext.matproj import MPRester
import pandas as pd
import os

#  ВСТАВЬ СЮДА СВОЙ API KEY
API_KEY = "PE4UJEOEpNrIPd29nrTUZzMs4YNvf9FA"

mpr = MPRester(API_KEY)

# создаём папку если нет
os.makedirs("sample", exist_ok=True)

# загружаем данные
df = df = pd.read_csv("train.csv")

ids = df["material_id"].unique()

print(f"Найдено {len(ids)} материалов")

for mid in ids:
    try:
        print(f"Скачиваю {mid}...")
        struct = mpr.get_structure_by_material_id(mid)
        struct.to(filename=f"sample/{mid}.cif")
    except Exception as e:
        print(f"Ошибка с {mid}: {e}")

print("ГОТОВО ✅")