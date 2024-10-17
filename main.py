# Telco Churn Prediction

# İş Problemi

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

# Veri Seti Hikayesi

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
# İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

# DEĞİŞKENLER
"""
CustomerId: Müşteri İd’si
Gender: Cinsiyet
SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
tenure: Müşterinin şirkette kaldığı ay sayısı
PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
TotalCharges: Müşteriden tahsil edilen toplam tutar
Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)
"""

# Görev 1 : Keşifçi Veri Analizi

# Kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Veri setinin import edilmesi
df_ = pd.read_csv("Case Study 3/TelcoChurn/Telco-Customer-Churn.csv")
df = df_.copy()

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
df.head()
df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
cat_but_car = [col for col in cat_but_car if col not in "customerID"]

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df[cat_but_car] = df[cat_but_car].apply(pd.to_numeric, errors='coerce')
df[num_but_cat] = df[num_but_cat].replace({1: "yes", 0: "no"}).astype("object")
df.info()

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

# Numerik Değişkenler
df.describe().T

# Kategorik Değişkenler
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, True)


# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_with_cat(dataframe, target, categorical_col):
    # Sayıları hesapla
    count_summary = dataframe.groupby(target).agg({categorical_col: "count"})

    # Yüzdeleri hesapla
    percentage_summary = dataframe.groupby(target).agg({categorical_col: lambda x: (x.count() / len(dataframe)) * 100})

    # Sonuçları birleştir
    summary = pd.concat([count_summary, percentage_summary.rename(columns={categorical_col: 'percentage'})], axis=1)
    summary.columns = ['count', 'percentage']

    print(summary, end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


# Adım 5: Aykırı gözlem var mı inceleyiniz.

for col in df.select_dtypes(exclude="object").columns:
    sns.boxplot(df[col])
    plt.show()

# Adım 6: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

# Görev 2: Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df[df["TotalCharges"].isna() == True] # 11 gözlem

df.groupby(["SeniorCitizen", "Dependents", "Contract", "Churn","gender"]).agg("TotalCharges").mean()["no", "Yes", slice(None), "No","Male"]
df.groupby(["SeniorCitizen", "Dependents", "Contract", "Churn","gender"]).agg("TotalCharges").mean()["no", "Yes", slice(None), "No","Female"]

# two year no female => 3202.545745
# two year no male => 3440.664516
# one year => 2448.786486


df.loc[(df["TotalCharges"].isnull()) & (df["gender"]=="Male") & (df["Contract"] == "Two year"), "TotalCharges"] = 3440.664516 # 5 gözlem
df.loc[(df["TotalCharges"].isnull()) & (df["gender"]=="Female") & (df["Contract"] == "Two year"), "TotalCharges"] = 3202.545745 # 5 gözlem
df.loc[(df["TotalCharges"].isnull())  & (df["Contract"] == "One year"), "TotalCharges"] = 2448.786486 # 1 gözlem

df.isnull().sum() # 0

# Adım 2: Yeni Değişkenler Oluşturunuz

df.head()
df.describe()

# Tenure değişkenini gruplara ayırma
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 1000000], labels=['Yeni', 'Orta', 'Uzun vadeli'])
df['TenureGroup'] = df['TenureGroup'].astype("object")

# Hizmetlerin "Evet" olduğu durumları sayarak ServiceCount değişkenini oluşturma
df['ServiceCount'] = df[['PhoneService', 'MultipleLines', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies']].apply(lambda x: x.eq('Yes').sum(), axis=1)
df["ServiceCount"] = df["ServiceCount"].astype(int)



df.info()
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
df.head()
# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()
df.info()
# One Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 7 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.info()
df.head()


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

numeric_cols = [col for col in df.columns if df[col].dtype not in ["object"]]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.head()

# Modelleme
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Feature Önemi
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

rf_final = RandomForestClassifier(random_state=12345).fit(X, y)
plot_importance(rf_final, X)

xgboost_final = XGBClassifier(random_state=12345).fit(X, y)
plot_importance(xgboost_final, X)

lgbm_final = LGBMClassifier(random_state=12345).fit(X, y)
plot_importance(lgbm_final, X)

catboost_final = CatBoostClassifier(verbose=False, random_state=12345).fit(X, y)
plot_importance(catboost_final, X)