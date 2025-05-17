import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from .data import *
import random
from datetime import datetime, timedelta

def generate_random_time(open_t, close_t):
    current_year = datetime.now().year
    random_month = random.randint(1, 12)
    random_day = random.randint(1, 28) 
    random_date = datetime(current_year, random_month, random_day)

    open_time = datetime.combine(random_date, datetime.strptime(open_t, "%H:%M").time())
    close_time = datetime.combine(random_date, datetime.strptime(close_t, "%H:%M").time())

    delta = close_time - open_time
    random_seconds = random.randint(0, delta.seconds)
    purchase_time = open_time + timedelta(seconds=random_seconds)

    return purchase_time.strftime('%Y-%m-%dT%H:%M:%S+03:00')

used_cards = defaultdict(int)
def generate_card_number():
    bank = random.choices(banks, weights=[sber, tenek, alf, vtb, psb], k=1)[0]
    bin_code = random.choice(bin_codes[bank])
    payment_system = random.choices(payment_systems, weights=[mir, visa, master], k=1)[0] 
    payment_system_code = payment_systems_codes[payment_system]
    card_number = ''.join(random.choices('0123456789', k=10))
    
    used_cards[card_number] += 1
    return f"{payment_system_code}{bin_code}{card_number}", bank, payment_system

def generate_unique_card_number():
    while True:
        card_number, bank, payment_system = generate_card_number()
        if used_cards[card_number] <= 5:  # Карта может быть использована до 5 раз
            return card_number, bank, payment_system


def generate_quantity_and_price(a):
    quantity = random.randint(1,10)
    price = 0
    for _ in range(1,quantity+1):    
        price += (random.randint( a[len(a)-2] , a[len(a)-1] ))
    return quantity, price

'''def z_c(flt): # zero_counter
    if flt != float:
        return 0
    count = 1
    while flt <=0 :
        flt*10
        count *= 10
    return count'''



#print("Введите вероятность для каждого банка , в сумме должно быть 1!")
sber = 0.34 #float(input(f"{'Сбребанк: '}" ))
tenek = 0.46 #float(input(f"{'Тинькофф: '}"))
alf = 0.1 #float(input(f"{'Альфа-Банк: '}"))
vtb = 0.05 #float(input(f"{'ВТБ: '}"))
psb = 0.05 #float(input(f"{'ПСБ: '}"))  
while (int(sber*100) + int(tenek*100) + int(alf*100) + int(vtb*100) + int(psb*100) > 100):
    print("Ошибка! Введите вероятность для каждого банка , в сумме должно быть 1!")
    sber = float(input(f"{'Сбребанк: '}" ))
    tenek = float(input(f"{'Тинькофф: '}"))
    alf = float(input(f"{'Альфа-Банк: '}"))
    vtb = float(input(f"{'ВТБ: '}"))
    psb = float(input(f"{'ПСБ: '}"))

#print("Введите вероятность для платёжных систем: ")
mir = 0.6 #float(input(f"{'MИР: '}"))
visa = 0.3 #float(input(f"{'Visa '}"))
master = 0.1 #float(input(f"{'MasterCard '}"))
while (int(mir*100) + int(visa*100) + int(master*100)) > 100:
    print("Ошибка! Введите вероятность для платёжных систем: ")
    mir = float(input(f"{'MИР: '}"))
    visa = float(input(f"{'Visa '}"))
    master = float(input(f"{'MasterCard '}"))

def generate_purchase_row():

    store = random.choice(stores)
    latitude, longitude, open_t, close_t = random.choice(stores_data[store]) 
    purchase_time = generate_random_time(open_t,close_t)
    category = random.choice(categories[store])  
    brand = random.choice(brands[category][ : (len(brands[category]) - 2)])  
    card_number, bank, payment_system = generate_unique_card_number()   
    quantity, price = generate_quantity_and_price(brands[category])
    
    return {
        "Магазин": store,
        "Широта": latitude,
        "Долгота": longitude,
        "Дата и время": purchase_time,
        "Категория": category,
        "Бренд": brand,
        "Номер карты": card_number,
        "Банк": bank,
        "Платежная система": payment_system,
        "Количество товаров": quantity,
        "Стоимость": price
    }

def generate_dataset(num_rows):
    dataset = []
    for _ in range(num_rows):
        dataset.append(generate_purchase_row())
    return pd.DataFrame(dataset)
