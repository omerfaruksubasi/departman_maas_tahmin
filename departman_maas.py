
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Veri setini yükleyelim
df = pd.read_csv("polynomial.csv", sep=";")

# Polinom derecesini belirleyelim
polynomial_regression = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])

# Lineer regresyon modelini oluşturalım ve eğitelim
reg = LinearRegression()
reg.fit(x_polynomial, df['maas'])

# Görselleştirme
#plt.scatter(df['deneyim'], df['maas'], color="blue", label="Veri")

# Polinom regresyonu çizelim
#x_range = range(int(min(df['deneyim'])), int(max(df['deneyim']))+1)#range(1,5) böyle yazdığımızda 1,2,3,4 ü kapsadığı için +1 yazdık
#y_head = reg.predict(polynomial_regression.transform(pd.DataFrame(x_range)))
#pd.DataFrame() fonksiyonu, farklı veri tiplerini (listeler, diziler, sözlükler vb.) içeren verileri bir DataFrame'e dönüştürmek için kullanılır.
#plt.plot(x_range, y_head, color="red", label="Polinom Regresyon")

# plt.legend()
# plt.xlabel("Deneyim")
# plt.ylabel("Maaş")
# plt.title("Polinom Regresyon ile Maaş Tahmini")
# plt.show()

deneyim = int(input("yeni çalışanın deneyim yılını giriniz: "))
yeni_veri = polynomial_regression.transform([[deneyim]])
tahmin = reg.predict(yeni_veri)
print("tahmini maaşı: ",tahmin)



