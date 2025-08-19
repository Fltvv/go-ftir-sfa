#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_notebooks.py — выполняет список Jupyter-ноутбуков:
- Все ноутбуки, кроме последних двух, запускаются последовательно.
- Последние два ноутбука запускаются ПАРАЛЛЕЛЬНО (если их действительно ≥2).
- Выполненные копии сохраняются в ./executed с суффиксом _executed.ipynb.

Зависимости: nbformat, nbconvert
Установка: pip install nbformat nbconvert
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
except ImportError as e:
    sys.stderr.write(
        "Требуются пакеты nbformat и nbconvert.\n"
        "Установите: pip install nbformat nbconvert\n"
    )
    raise

# === УКАЖИТЕ СВОИ НОУТБУКИ В НУЖНОМ ПОРЯДКЕ ===
NOTEBOOKS = [
    "code/03a_train_test_split.ipynb",
    "code/03b_feature_extraction.ipynb",
    "code/03c_feature_engineering.ipynb",
    "code/04a_modeling_lr.ipynb",
    "code/04b_modeling_lr.ipynb"
]

# === НАСТРОЙКИ ВЫПОЛНЕНИЯ ===
KERNEL_NAME = "python3"   # имя ядра (при необходимости измените)
TIMEOUT_SEC = 3600        # таймаут на ноутбук; 0 — без ограничения
ALLOW_ERRORS = False      # False => падаем при первой ошибке
OUT_DIRNAME = "executed"  # куда сохранять выполненные ноутбуки

def run_notebook(nb_path: Path, out_dir: Path) -> Path:
    """
    Выполняет ноутбук nb_path и сохраняет выполненную копию в out_dir.
    Возвращает путь к сохранённому файлу.
    """
    if not nb_path.exists():
        raise FileNotFoundError(f"Не найден ноутбук: {nb_path}")

    print(f"\n=== Запуск: {nb_path} ===")
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Важно: рабочая директория — папка ноутбука (для относительных путей)
    resources = {"metadata": {"path": str(nb_path.parent)}}

    ep = ExecutePreprocessor(
        timeout=TIMEOUT_SEC,
        kernel_name=KERNEL_NAME,
        allow_errors=ALLOW_ERRORS,
    )

    t0 = time.time()
    try:
        ep.preprocess(nb, resources=resources)
    except CellExecutionError as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        failed_path = out_dir / f"{nb_path.stem}_FAILED.ipynb"
        with failed_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"❌ Ошибка: {nb_path}\nЧастично выполненный ноутбук сохранён: {failed_path}")
        raise

    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nb_path.stem}_executed.ipynb"
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Готово: {out_path} (время: {elapsed:.1f} с)")
    return out_path

def main():
    if not NOTEBOOKS:
        print("Список NOTEBOOKS пуст — добавьте пути к .ipynb.")
        sys.exit(2)

    base_dir = Path.cwd()
    out_dir = base_dir / OUT_DIRNAME

    # Разделяем: всё до последних двух — последовательно; последние два — параллельно
    seq_part = NOTEBOOKS[:-2] if len(NOTEBOOKS) >= 3 else NOTEBOOKS[:-2]  # будет [] если <3
    par_part = NOTEBOOKS[-2:] if len(NOTEBOOKS) >= 2 else NOTEBOOKS[-2:]  # возьмет 1 или 2

    started = time.time()

    # 1) Последовательное выполнение
    for nb in seq_part:
        nb_path = (base_dir / nb).resolve() if not Path(nb).is_absolute() else Path(nb)
        try:
            run_notebook(nb_path, out_dir)
        except Exception as e:
            print(f"\n⛔ Остановлено из-за ошибки при выполнении: {nb_path}\n{e}")
            sys.exit(1)

    # 2) Параллельное выполнение последних двух (или одного, если в списке всего 1–2)
    if par_part:
        print("\n=== Параллельный запуск последних ноутбуков ===")
        futures = {}
        with ProcessPoolExecutor(max_workers=min(2, len(par_part))) as ex:
            for nb in par_part:
                nb_path = (base_dir / nb).resolve() if not Path(nb).is_absolute() else Path(nb)
                futures[ex.submit(run_notebook, nb_path, out_dir)] = nb_path

            had_error = False
            for fut in as_completed(futures):
                nb_path = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"\n⛔ Ошибка при параллельном выполнении: {nb_path}\n{e}")
                    had_error = True
        if had_error and not ALLOW_ERRORS:
            sys.exit(1)

    total = time.time() - started
    print(f"\n🎉 Готово. Итоговое время: {total:.1f} с\nВыполненные ноутбуки: {out_dir}")

if __name__ == "__main__":
    main()
