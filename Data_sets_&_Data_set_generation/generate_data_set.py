from Data_set_generation.data_generation import *
from Data_set_generation.validators import validate_dataset

print("Введите число запросов (строк в датасете)")

num_rows = int(input())
while not num_rows.is_integer :
    print("Допущена ошибка! Попытайтесь снова")
    num_rows = int(input())

dataset = generate_dataset(num_rows)

validate_dataset(dataset)

dataset.to_excel(f"dataset_{num_rows}.xlsx", index=False)
#dataset.to_csv('dataset.csv', index=False)

print(f"Датасет успешно сгенерирован и сохранен. Всего строк: {len(dataset)}")
print(dataset)