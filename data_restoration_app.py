import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

class DataRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Инструмент восстановления данных")
        self.root.geometry("500x650")
        
        self.source_path = None
        self.gaps_path = None
        self.original_path = None
        self.restored_path = None
        
        self.setup_ui()
        self.root.bind('<Return>', lambda event: self.remove_data())

    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.control_panel = ttk.Frame(main_container, padding=10, relief="groove")
        self.control_panel.pack(fill=tk.X)
        
        self.output_panel = ttk.Frame(main_container, padding=10, relief="groove")
        self.output_panel.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        """Панель управления"""
        ttk.Label(self.control_panel, text="Исходный файл:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.source_entry = ttk.Entry(self.control_panel, state='readonly')
        self.source_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(self.control_panel, text="Выбрать", command=self.select_source).grid(row=0, column=2, sticky=tk.W, pady=2)

        ttk.Label(self.control_panel, text="Удалить (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.percent_entry = ttk.Entry(self.control_panel)
        self.percent_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        ttk.Button(self.control_panel, text="Удалить данные", command=self.remove_data).grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(self.control_panel, text="Файл для восстановления:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.gaps_entry = ttk.Entry(self.control_panel, state='readonly')
        self.gaps_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(self.control_panel, text="Выбрать", command=self.select_gaps).grid(row=3, column=2, sticky=tk.W, pady=2)

        ttk.Label(self.control_panel, text="Метод восстановления:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.method_combo = ttk.Combobox(self.control_panel, state='readonly')
        self.method_combo['values'] = ('Хот-Дек', 'Заполнения значением медианы', 'Сплайн-интерполяция')
        self.method_combo.grid(row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        self.method_combo.current(0)
        ttk.Button(self.control_panel, text="Восстановить данные", command=self.restore_data).grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(self.control_panel, text="Исходный файл:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.original_entry = ttk.Entry(self.control_panel, state='readonly')
        self.original_entry.grid(row=6, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(self.control_panel, text="Выбрать", command=self.select_original).grid(row=6, column=2, sticky=tk.W, pady=2)

        ttk.Label(self.control_panel, text="Восстановленный файл:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.restored_entry = ttk.Entry(self.control_panel, state='readonly')
        self.restored_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(self.control_panel, text="Выбрать", command=self.select_restored).grid(row=7, column=2, sticky=tk.W, pady=2)
        ttk.Button(self.control_panel, text="Рассчитать погрешность", command=self.calculate_accuracy).grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)

        for i in range(9):
            self.control_panel.rowconfigure(i, weight=1)
        self.control_panel.columnconfigure(1, weight=1)

        """Панель вывода"""
        self.output_text = tk.Text(self.output_panel, wrap=tk.WORD, height=18, width=40)
        scrollbar = ttk.Scrollbar(self.output_panel, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def select_source(self):
        path = filedialog.askopenfilename(filetypes=[("Excel файлы", "*.xlsx")])
        if path:
            self.source_path = path
            self.update_entry(self.source_entry, os.path.basename(path))
   
    def select_gaps(self):
        path = filedialog.askopenfilename(filetypes=[("Excel файлы", "*.xlsx")])
        if path:
            self.gaps_path = path
            self.update_entry(self.gaps_entry, os.path.basename(path))
            
    def select_original(self):
        path = filedialog.askopenfilename(filetypes=[("Excel файлы", "*.xlsx")])
        if path:
            self.original_path = path
            self.update_entry(self.original_entry, os.path.basename(path))
            
    def select_restored(self):
        path = filedialog.askopenfilename(filetypes=[("Excel файлы", "*.xlsx")])
        if path:
            self.restored_path = path
            self.update_entry(self.restored_entry, os.path.basename(path))

    def update_entry(self, entry, text):
        entry.config(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, text)
        entry.config(state="readonly")

    def remove_data(self):
        if not self.source_path:
            messagebox.showerror("Ошибка", "Выберите исходный файл")
            return
        
        try:
            percent = float(self.percent_entry.get())
            if not 0 < percent < 50:
                raise ValueError
        except:
            messagebox.showerror("Ошибка!", "Поле заполнено некорректно! (введите от 0 до 50)")
            return

        df = pd.read_excel(self.source_path)
        arr = df.values.copy()
        rows, cols = arr.shape
        total = rows * cols
        
        mask = np.zeros_like(arr, dtype=bool)
        removed = 0
        block_size = 4
        
        positions = [(i,j) for i in range(rows-block_size+1) for j in range(cols-block_size+1)]
        np.random.shuffle(positions)
        
        for i,j in positions:
            if removed + block_size**2 > total*percent/100:
                break
            if not mask[i:i+block_size, j:j+block_size].any():
                mask[i:i+block_size, j:j+block_size] = True
                removed += block_size**2
        
        remaining = int(total*percent/100) - removed
        if remaining > 0:
            candidates = np.argwhere(~mask)
            np.random.shuffle(candidates)
            for idx in candidates[:remaining]:
                mask[tuple(idx)] = True
        
        arr[mask] = np.nan
        save_path = self.source_path.replace(".xlsx", f"_nan_{int(percent)}.xlsx")
        pd.DataFrame(arr, columns=df.columns).to_excel(save_path, index=False)
        messagebox.showinfo(
            "Готово",
            f"Файл с пропусками сохранён:\n{save_path}\n\n"
            f"Удалено {(removed + remaining)//11} из {total//11} ячеек ({(removed + remaining)/total*100:.2f}%)"
        )

    def restore_data(self):
        if not self.gaps_path:
            messagebox.showerror("Ошибка", "Выберите файл с пропусками")
            return
        
        method = self.method_combo.get()
        df = pd.read_excel(self.gaps_path)
        
        if method == 'Хот-Дек':
            name = 'hot-deck'
            restored_df = self.hot_deck_imputation(df)
        elif method == 'Заполнения значением медианы':
            name = 'median'
            restored_df = self.median_imputation(df)
        elif method == 'Сплайн-интерполяция':
            name = 'spline'
            restored_df = self.spline_interpolation(df)
        
        save_path = self.gaps_path.replace(".xlsx", f"_restored_{name}.xlsx")
        restored_df.to_excel(save_path, index=False)
        messagebox.showinfo("Готово", f"Восстановленный файл сохранён:\n{save_path}")
            
    def hot_deck_imputation(self, df):
        encoded_df = df.copy()
        encoders = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        # Частотное кодирование категориальных признаков
        for col in categorical_cols:
            freq = df[col].value_counts(normalize=True)
            encoded_df[col] = df[col].map(freq)
            encoders[col] = freq

        for col in numeric_cols:
            if encoded_df[col].notna().sum() > 0:
                encoded_df[col] = encoded_df[col].fillna(encoded_df[col].median())
        for col in categorical_cols:
            if encoded_df[col].notna().sum() > 0:
                encoded_df[col] = encoded_df[col].fillna(encoded_df[col].median())

        # Кластеризация датасета
        n_clusters = max(10, len(df) // 300)  # ~300 строк на кластер
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(encoded_df)

        missing_rows = df[df.isna().any(axis=1)].index

        def process_cluster(cluster_id, cluster_indices):
            changes = []
            cluster_df = encoded_df.iloc[cluster_indices]
            cluster_orig_df = df.iloc[cluster_indices]
            cluster_missing = cluster_orig_df.index.intersection(missing_rows)

            if len(cluster_df) == 0:
                return changes

            nn = NearestNeighbors(n_neighbors=min(200, max(1, len(cluster_df))), metric='nan_euclidean')
            nn.fit(cluster_df)

            for idx in cluster_missing:
                nan_cols = cluster_orig_df.columns[cluster_orig_df.loc[idx].isna()]
                query = encoded_df.loc[[idx]]
                distances, local_indices = nn.kneighbors(query)
                neighbor_indices = cluster_indices[local_indices[0]]

                for col in nan_cols:
                    neighbor_values = df.iloc[neighbor_indices][col].dropna()
                    if len(neighbor_values) > 0:
                        if col in categorical_cols:
                            changes.append({'index': idx, 'column': col, 'value': neighbor_values.mode()[0]})
                        else:
                            changes.append({'index': idx, 'column': col, 'value': neighbor_values.median()})
                    else:
                        closest_idx = neighbor_indices[0]
                        closest_value = df.iloc[closest_idx][col]
                        if not pd.isna(closest_value):
                            changes.append({'index': idx, 'column': col, 'value': closest_value})
            return changes

        # Параллельная бработка кластеров
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(process_cluster)(cluster_id, np.where(clusters == cluster_id)[0])
            for cluster_id in range(n_clusters)
        )

        for changes in results:
            for change in changes:
                df.loc[change['index'], change['column']] = change['value']

        for col in df.columns:
            if df[col].isna().any():
                non_null = df[col].dropna().values
                if len(non_null) > 0:
                    df[col] = df[col].apply(lambda x: np.random.choice(non_null) if pd.isna(x) else x)

        return df

    def median_imputation(self, df):
        df_filled = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if df_filled[col].notna().sum() > 0:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

        return df_filled

    def spline_interpolation(self, df):
        """Сплайн-интерполяция для числовых и категориальных данных с кодированием"""
        df_interp = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        encoders = {}
        for col in categorical_cols:
            freq = df_interp[col].value_counts(normalize=True)
            df_interp[col] = df_interp[col].map(freq)
            encoders[col] = freq

        for col in numeric_cols.union(categorical_cols):
            if df_interp[col].notna().sum() >= 2:
                df_interp[col] = df_interp[col].interpolate(method='spline', order=3)

        for col in categorical_cols:
            freq_values = encoders[col].values
            df_interp[col] = df_interp[col].apply(lambda x: min(freq_values, key=lambda v: abs(v - x)))
            df_interp[col] = df_interp[col].map({v: k for k, v in encoders[col].items()})

        return df_interp

    def calculate_accuracy(self):
        """Расчет погрешности между исходным и восстановленным файлами"""
        if not self.original_path or not self.restored_path:
            messagebox.showerror("Ошибка", "Выберите исходный и восстановленный датасет")
            return

        df_original = pd.read_excel(self.original_path)
        df_restored = pd.read_excel(self.restored_path)

        categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df_original.select_dtypes(include=['number']).columns

        numeric_errors = []
        categorical_errors = []
        self.output_text.delete(1.0, tk.END)

        for col in numeric_cols:
            error_sum = 0
            count = 0
            for orig, rest in zip(df_original[col], df_restored[col]):
                if pd.isna(orig):
                    continue
                count += 1
                if pd.isna(rest):
                    error_sum += 1
                elif orig != 0:
                    error_sum += abs(orig - rest) / abs(orig)
            if count > 0:
                error_percent = (error_sum / count) * 100
                numeric_errors.append(error_percent)
                self.output_text.insert(tk.END, f"Столбец '{col}' (числовой): {error_percent:.2f}%\n")
            else:
                self.output_text.insert(tk.END, f"Столбец '{col}' (числовой): Нет данных для расчета\n")

        for col in categorical_cols:
            errors = 0
            total = 0
            for orig, rest in zip(df_original[col], df_restored[col]):
                if pd.isna(orig):
                    continue
                total += 1
                if pd.isna(rest) or orig != rest:
                    errors += 1
            if total > 0:
                error_percent = (errors / total) * 100
                categorical_errors.append(error_percent)
                self.output_text.insert(tk.END, f"Столбец '{col}' (категориальный): {error_percent:.2f}%\n")
            else:
                self.output_text.insert(tk.END, f"Столбец '{col}' (категориальный): Нет данных для расчета\n")

        numeric_avg = sum(numeric_errors) / len(numeric_errors) if numeric_errors else 0
        categorical_avg = sum(categorical_errors) / len(categorical_errors) if categorical_errors else 0

        self.output_text.insert(tk.END, "\nИтог:\n")
        self.output_text.insert(tk.END, f"Средняя ошибка для числовых данных: {numeric_avg:.2f}%\n")
        self.output_text.insert(tk.END, f"Средняя ошибка для категориальных данных: {categorical_avg:.2f}%\n")
        overall_error = (numeric_avg + categorical_avg) / 2 if numeric_errors and categorical_errors else numeric_avg or categorical_avg
        self.output_text.insert(tk.END, f"Общая ошибка: {overall_error:.2f}%\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataRestorationApp(root)
    root.mainloop()