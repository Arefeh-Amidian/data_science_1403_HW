# %% [markdown]
# # **Project1**

# %% [markdown]
# ### *Arefeh Amidyan: 12/16/2024*

# %% [markdown]
# # **Import Libraries**

# %%
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# %%
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# %% [markdown]
# گاهی اوقات اگر به جای این دستور، فقط از 

# %% [markdown]
# !pip install fastparquet 

# %% [markdown]
# استفاده کنید، کتابخانه ممکن است در نسخه پایتونی دیگری نصب شود، و باعث شود کد شما نتواند آن را پیدا کند

# %% [markdown]
# 
# این روش سازگاری با محیط جاری را تضمین می‌کند

# %%
# !where python
# !pip install matplotlib
import sys
# !{sys.executable} -m pip install matplotlib
# !{sys.executable} -m pip install seaborn
# !pip install pyarrow
# !pip install fastparquet
# !{sys.executable} -m pip install pyarrow
# !{sys.executable} -m pip install fastparquet
# pip install pandas matplotlib seaborn sqlalchemy
!{sys.executable} -m pip install sqlalchemy





# %% [markdown]
# # ***Load Data***

# %%
home_team_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/home_team"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/home_team/" + file)
    home_team_df = pd.concat([home_team_df, single_stats], axis='rows', ignore_index=True)

away_team_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/away_team"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/away_team/" + file)
    away_team_df = pd.concat([away_team_df, single_stats], axis='rows', ignore_index=True)


# %% [markdown]
# # **Analysis Question 1:**

# %% [markdown]
# # **1. How many tennis players are included in the dataset?**
# 

# %% [markdown]
# # Combine the dataframes

# %%
combined_df = pd.concat([home_team_df, away_team_df], axis=0, ignore_index=True)

# %%
combined_df

# %%
combined_df.shape

# %%
combined_df.head()

# %%
combined_df.info()

# %%
combined_df.describe()

# %% [markdown]
# ## Check for Missing Values

# %%
print("Missing Values Before Handling:")
print(combined_df.isnull().sum())

# %%
missing_percentage = combined_df.isnull().mean() * 100
print(missing_percentage)

# %% [markdown]
# ## Convert Datatypes of columns

# %%
# تبدیل ستون‌های عددی  
combined_df['height'] = combined_df['height'].astype(float)  

# برای weight، ابتدا بررسی کنید که آیا می‌توان آن را به float تبدیل کرد  
combined_df['weight'] = pd.to_numeric(combined_df['weight'], errors='coerce')  

# برای current_prize و total_prize، ابتدا بررسی کنید که آیا می‌توان آن‌ها را به float تبدیل کرد  
combined_df['current_prize'] = pd.to_numeric(combined_df['current_prize'], errors='coerce')  
combined_df['total_prize'] = pd.to_numeric(combined_df['total_prize'], errors='coerce')  

# برای current_rank، ابتدا بررسی کنید که آیا می‌توان آن را به int تبدیل کرد  
combined_df['current_rank'] = pd.to_numeric(combined_df['current_rank'], errors='coerce')  

# برای turned_pro، به int تبدیل کنید  
combined_df['turned_pro'] = pd.to_numeric(combined_df['turned_pro'], errors='coerce')  

# بررسی نوع داده‌ها بعد از تبدیل  
print(combined_df.dtypes)

# %%
import seaborn as sns  
import matplotlib.pyplot as plt  

sns.histplot(combined_df['height'], bins=30, kde=True)  # kde نمایانگر تابع چگالی هسته‌ای است.  
plt.title('Histogram of Height')  
plt.xlabel('Height')  
plt.ylabel('Frequency')  
plt.show()

# %% [markdown]
# # Analyze outliers in numeric columns (e.g., height)

# %%
sns.boxplot(combined_df['height'])  
plt.title('Box Plot of Height')  
plt.show()

# %% [markdown]
# میانگین معمولاً برای داده‌هایی که توزیع نرمال دارند مناسب است.
# 

# %% [markdown]
# اگر داده‌ها دورافتاده داشتند، بهتر است به جای میانگین از میانه (median) استفاده کنید.

# %% [markdown]
# این روش مناسب است، چون داده‌های متنی معمولاً ارزش عددی ندارند و مد (پرکاربردترین مقدار) بهترین نماینده برای مقادیر گم‌شده است

# %% [markdown]
# بررسی جامع تکراری‌ها
# شما بررسی تکراری‌ها را انجام داده‌اید، ولی فقط در سطح کلیدها یا ستون‌های خاص. این خوب است، اما بهتر است یک بار کل مجموعه داده را بررسی کنید:

# %%
total_duplicates = combined_df.duplicated().sum()
print(f'Total number of duplicates are {total_duplicates}')


# %% [markdown]
# # **Handling Missing Values**

# %% [markdown]
# # Replace missing values based on appropriate methods

# %% [markdown]
# برای ستون‌های عددی، به‌صورت منطقی میانگین یا میانه را انتخاب کردیم.

# %% [markdown]
# ### چرا روش **میانه (Median)** 
# 
# #### 1. **میانگین (Mean):**
# - میانگین به دلیل حساسیت به مقادیر دورافتاده ممکن است دچار انحراف شود.
# - برای مثال، اگر مقادیر غیرمعمولی مانند 300 سانتی‌متر یا 50 سانتی‌متر در ستون `height` وجود داشته باشد، میانگین به سمت این مقادیر کشیده خواهد شد و نماینده دقیقی از توزیع داده‌ها نخواهد بود.
# 
# #### 2. **میانه (Median):**
# - میانه، مقدار وسط داده‌ها است و تحت تأثیر مقادیر دورافتاده قرار نمی‌گیرد.
# - این ویژگی، میانه را به انتخاب بهتری برای ستون‌هایی مانند `height` و `weight` تبدیل می‌کند که ممکن است مقادیر دورافتاده داشته باشند.
# 
# #### 📌 **نتیجه‌گیری:**
# - برای ستون‌هایی که ممکن است داده‌های غیرعادی (Outliers) داشته باشند، استفاده از **میانه** به جای **میانگین** منطقی‌تر است. این روش باعث می‌شود توزیع داده‌ها بهتر نمایش داده شود.
# 

# %%
combined_df['height'].fillna(combined_df['height'].median(), inplace=True)  # Using median due to possible outliers
combined_df['weight'].fillna(combined_df['weight'].mean(), inplace=True)   # Assuming no significant outliers in weight
combined_df['current_prize'].fillna(combined_df['current_prize'].mean(), inplace=True)
combined_df['total_prize'].fillna(combined_df['total_prize'].mean(), inplace=True)
combined_df['current_rank'].fillna(combined_df['current_rank'].median(), inplace=True)
combined_df['turned_pro'].fillna(combined_df['turned_pro'].median(), inplace=True)

# %% [markdown]
# # Fill missing categorical columns with mode

# %% [markdown]
# برای ستون‌های متنی، از مد استفاده کردیم چون منطقی‌ترین جایگزین برای داده‌های متنی است.

# %%
combined_df['residence'].fillna(combined_df['residence'].mode()[0], inplace=True)
combined_df['birthplace'].fillna(combined_df['birthplace'].mode()[0], inplace=True)
combined_df['plays'].fillna(combined_df['plays'].mode()[0], inplace=True)
combined_df['country'].fillna(combined_df['country'].mode()[0], inplace=True)

# %% [markdown]
# اگر دورافتاده‌های شناسایی‌شده منطقی نباشند (مانند قد غیرممکن 300 سانتی‌متر)، می‌توان آن‌ها را مستقیماً حذف کرد.
# 
# برای حذف مقادیر دورافتاده، می‌توان از محدوده بین چارکی (IQR) استفاده کرد:

# %%
# Q1 = combined_df['height'].quantile(0.25)
# Q3 = combined_df['height'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # حذف مقادیر دورافتاده
# combined_df = combined_df[(combined_df['height'] >= lower_bound) & (combined_df['height'] <= upper_bound)]


# %% [markdown]
# این کد داده‌ها را محدود به محدوده منطقی کرده و دورافتاده‌ها را حذف می‌کند

# %% [markdown]
# # Check for Missing Values After Handling

# %%
print("Missing Values After Handling:")
print(combined_df.isnull().sum())

# %% [markdown]
# # Check for Outliers in Updated Data

# %%
sns.boxplot(x=combined_df['height'])
plt.title("Height Boxplot After Handling Missing Values")
plt.show()

# %% [markdown]
# # Check for Duplicates

# %%
total_duplicates = combined_df.duplicated().sum()
print(f'Total number of duplicates before removal: {total_duplicates}')

# %% [markdown]
# # Drop Duplicates

# %%
combined_df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {combined_df.shape}")

# %% [markdown]
# # Verify No Duplicates Remain

# %%
print(f"Total number of duplicates after removal: {combined_df.duplicated().sum()}")

# %% [markdown]
# # Extract Unique Players

# %%
unique_players_count = combined_df['player_id'].nunique()
print("Total Number of Unique Players:", unique_players_count)

# %%
# تعداد بازیکنان بر اساس جنسیت
gender_counts = combined_df['gender'].value_counts()
plt.tick_params(colors='gray')

# رسم نمودار با رنگ‌های ملیح
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="pastel")  # استفاده از پالت pastel
plt.title("Number of Tennis Players by Gender", color="gray", loc="left")
plt.xlabel("Gender", color="gray", loc="left")
plt.ylabel("Count", color="gray", loc="bottom")
ax = plt.gca()
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['top'].set_color('gray')
# حذف حاشیه‌های راست و بالا
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.show()


# %% [markdown]
# # **2. What is the average age of the players?**

# %% [markdown]
# **Can't answer this question because of non enough data**

# %%
from datetime import datetime

# بررسی وجود ستون birth_date
if 'birth_date' in combined_df.columns:
    # تبدیل ستون به datetime
    combined_df['birth_date'] = pd.to_datetime(combined_df['birth_date'], errors='coerce')
    
    # محاسبه سن بازیکنان
    current_date = datetime.now()
    combined_df['age'] = combined_df['birth_date'].apply(lambda x: current_date.year - x.year if pd.notnull(x) else None)
    
    # محاسبه میانگین سن
    average_age = combined_df['age'].mean()
    print(f"The average age of the players is: {average_age:.2f} years")
else:
    print("The dataset does not contain a 'birth_date' column.")


# %% [markdown]
# # **3. Which player has the highest number of wins?**

# %% [markdown]
# ### ***Load another data we want***

# %%
event_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/event"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/event/" + file)
    event_df = pd.concat([event_df, single_stats], axis= 'rows', ignore_index=True)

# %%
event_df.head()

# %%
event_df.shape

# %%
event_df.info()

# %%
event_df.describe()

# %% [markdown]
# ### ***Check missing values***

# %%
missing_values = event_df.isnull().sum()
missing_values

# %%
missing_percentage = event_df.isnull().mean() * 100
print(missing_percentage)

# %% [markdown]
# ### ***remove non-numeric values from the columns home_team_seed and away_team_seed***

# %%
event_df['home_team_seed'] = pd.to_numeric(event_df['home_team_seed'], errors='coerce')
event_df['away_team_seed'] = pd.to_numeric(event_df['away_team_seed'], errors='coerce')


# %%
event_df.isnull().sum()

# %%
event_df.dropna(subset=['winner_code'], inplace=True)


# %% [markdown]
# ## **Filling missing values on columns**

# %%
# event_df['winner_code'].fillna(event_df['winner_code'].mode()[0], inplace=True)
event_df['first_to_serve'].fillna(event_df['first_to_serve'].mode()[0], inplace=True)
event_df['home_team_seed'].fillna(event_df['home_team_seed'].median(), inplace=True)
event_df['away_team_seed'].fillna(event_df['away_team_seed'].median(), inplace=True)
event_df['start_datetime'].fillna(event_df['start_datetime'].mode()[0], inplace=True)


# %%
combined_df.isnull().sum()

# %% [markdown]
# ## **Count win of home and away teams**

# %%
host_wins = event_df[event_df['winner_code'] == 1].shape[0]
guest_wins = event_df[event_df['winner_code'] == 2].shape[0]

host_wins, guest_wins


# %% [markdown]
# ## **Extract winner player home & away team**

# %%
#home player winner
home_wins = event_df[event_df['winner_code'] == 1]

# away player winner
away_wins = event_df[event_df['winner_code'] == 2]
home_winners = home_wins.merge(home_team_df, on='match_id')['player_id']
away_winners = away_wins.merge(away_team_df, on='match_id')['player_id']

# %% [markdown]
# ## **merge a list of winners and count the number of wins for each player**

# %%
all_winners = pd.concat([home_winners, away_winners])
win_counts = all_winners.value_counts()


# %% [markdown]
# ## **Find the players with the most wins**

# %%
max_winner = win_counts.max()
top_players = win_counts[win_counts == max_winner].index

# %% [markdown]
# ## **Draw a bar chart for the top 10 players based on the number of wins**

# %%
top_10_players = win_counts.head(10)
top_10_players_with_names = top_10_players.reset_index().merge(combined_df, on='player_id')
top_10_players_with_names
top_10_players_with_names = top_10_players_with_names.set_index('name')

# تعریف پالت رنگی
palette = sns.color_palette("Blues_r", 10)

# رسم نمودار میله‌ای عمودی
plt.figure(figsize=(14, 6))  
plt.tick_params(colors='gray')

sns.barplot(x=top_10_players_with_names.index, y=top_10_players_with_names['count'], palette=palette, orient="v")
plt.title('Top 10 Players with Most Wins', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.xlabel('Player Names', fontsize=12, color='gray', loc='left')
plt.ylabel('Win Count', fontsize=12, color='gray', loc='bottom')
plt.xticks(rotation=0, ha="right")  
ax = plt.gca()
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['top'].set_color('gray')
# حذف حاشیه‌های راست و بالا
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()


# %%
# ساخت دیتافریم نهایی
top_10_players_with_names_table = top_10_players.reset_index().merge(combined_df[['player_id', 'name']], on='player_id')

# حذف تکرار اسامی بازیکنان و تنظیم نمایه
top_10_players_with_names_table = top_10_players_with_names_table.drop_duplicates(subset=['player_id']).set_index('name')

# مرتب‌سازی بر اساس تعداد برد
top_10_players_with_names_table = top_10_players_with_names_table.sort_values(by='count', ascending=False)

# نمایش جدول نهایی
print(top_10_players_with_names_table)
# top_10_players_with_names


# %% [markdown]
# # **4. What is the longest match recorded in terms of duration?**
# 

# %%
time_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/time"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/time/" + file)
    time_df = pd.concat([time_df, single_stats], axis= 'rows', ignore_index=True)

# %%
time_df.head()

# %%
time_df.info()

# %% [markdown]
# ## **Missing values**

# %%
time_df.isnull().sum()

# %% [markdown]
# ## **Handle Missing values**

# %%
# پر کردن مقادیر نال با 0
time_df.fillna(0, inplace=True)

# %% [markdown]
# ## **Convert types of columns**

# %%
# تبدیل مقادیر ستون‌های دوره‌ها به نوع عددی (در صورت نیاز)
for col in ['period_1', 'period_2', 'period_3', 'period_4', 'period_5']:
    time_df[col] = pd.to_numeric(time_df[col], errors='coerce')

# %%
time_df.isnull().sum()

# %% [markdown]
# ## **Longest match**

# %%
# محاسبه مجموع دوره‌ها (مدت زمان کل مسابقه) برای هر مسابقه
time_df['total_duration'] = (
    time_df['period_1'] +
    time_df['period_2'] +
    time_df['period_3'] +
    time_df['period_4'] +
    time_df['period_5']
)

# تبدیل total_duration از ثانیه به دقیقه
time_df['total_duration'] = time_df['total_duration'] / 60

# پیدا کردن طولانی‌ترین مسابقه
longest_match = time_df.loc[time_df['total_duration'].idxmax()]

# استخراج اطلاعات طولانی‌ترین مسابقه
longest_match_id = longest_match['match_id']
longest_match_duration = longest_match['total_duration']

print(f"longest_match_id: {longest_match_id} with match time of: {longest_match_duration:.2f} min")


# %% [markdown]
# # **5. How many sets are typically played in a tennis match?**
# 

# %%
time_df.head(10)

# %%
time_df.isnull().sum()

# %%
# حذف سطرهایی که period_1 برابر با صفر است
time_df = time_df[time_df['period_1'] != 0]


# محاسبه تعداد ست‌ها
#بررسی وجود مقدار در هر پریود 
time_df['sets_played'] = time_df[['period_1', 'period_2', 'period_3']].gt(0).sum(axis=1)
time_df

# %%
time_df.drop(columns=['period_4', 'period_5'])

# %%
# تعداد تکرار آن عدد
most_frequent_sets = time_df['sets_played'].mode()[0]

# نمایش نتایج
print("sets are typically played in a tennis match:", most_frequent_sets)

# %%
# محاسبه تعداد ست‌ها و درصد آنها
sets_counts = time_df['sets_played'].value_counts()
sets_labels = [f"{int(label)} Sets" for label in sets_counts.index]

# رسم نمودار دایره‌ای
plt.figure(figsize=(8, 8))
colors = ['#ffe6e6', '#cce5ff', '#e6ffe6', '#fff2e6']  # هلویی بسیار روشن، آبی بسیار روشن، سبز بسیار روشن، کرمی بسیار روشن
explode = [0.05] * len(sets_counts)  # جدا کردن بخش‌ها
total_matches = len(time_df)

plt.pie(
    sets_counts,
    labels=sets_labels,
    autopct=lambda pct: f"{int(pct/100 * total_matches)}\n({pct:.1f}%)",
    startangle=140,
    colors=colors,
    explode=explode,
    textprops={'color': 'gray'},  
    wedgeprops={'edgecolor': 'gray'}
)

# عنوان نمودار
plt.title(' Typically Sets Played in a Tennis Matches', fontsize=16, color='gray', fontweight='bold')

# نمایش نمودار
plt.tight_layout()
plt.show()


# %% [markdown]
# # **6. Which country has produced the most successful tennis players?**

# %%
# استخراج بازیکنانی که در تیم خانه برنده شده‌اند به همراه کشورشان
home_winners = home_wins.merge(home_team_df[['match_id', 'player_id', 'country']], on='match_id', how='left')

# استخراج بازیکنانی که در تیم میهمان برنده شده‌اند به همراه کشورشان
away_winners = away_wins.merge(away_team_df[['match_id', 'player_id', 'country']], on='match_id', how='left')

# ادغام لیست برندگان تیم خانه و تیم میهمان
all_winners = pd.concat([home_winners, away_winners])

# شمارش تعداد بردهای هر کشور
country_win_counts = all_winners['country'].value_counts()

# انتخاب 10 کشور برتر
top_countries = country_win_counts.head(10)

# ساخت پالت سفارشی شبیه به `rocket` ولی با طیف سبز
custom_palette = sns.light_palette("Gold", n_colors=len(top_countries), reverse=True)

# رسم نمودار میله‌ای افقی برای 10 کشور برتر
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette=custom_palette)  

# عنوان و برچسب‌های نمودار
plt.title('Top 10 Countries with Most Tennis Wins', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.xlabel('Number of Wins', fontsize=10, color='gray', loc="left")
plt.ylabel('Countries', fontsize=10, color='gray', loc="bottom")

# تغییر رنگ اعداد و خطوط مدرج به خاکستری
plt.tick_params(colors='gray')
ax = plt.gca()
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')

# حذف حاشیه‌های راست و بالا
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# نمایش نمودار
plt.show()


# %%
country_win_counts.head(10)

# %%
country_win_counts.tail()

# %% [markdown]
# # **7. What is the average number of aces per match?**

# %%
# Load and combine all statistics parquet files into a single DataFrame
statistics_files = "tennis_data_20231212/raw/raw_statistics_parquet"
statistics_df = pd.concat(
    [pd.read_parquet(os.path.join(statistics_files, file)) for file in os.listdir(statistics_files)],
    ignore_index=True
)

# %%
statistics_df.head()

# %%
statistics_df.info()

# %%
statistics_df.describe()

# %%
statistics_df.isnull().sum()

# %%
total_duplicates = statistics_df.duplicated().sum()
print(f'Total number of duplicates are {total_duplicates}')

# %%
statistics_df['home_total'].fillna(statistics_df['home_total'].median(), inplace=True)  
statistics_df['away_total'].fillna(statistics_df['away_total'].median(), inplace=True)  

# %%
# statistics_df.dropna(subset=['home_total', 'away_total'], inplace=True)

# %%
print("Missing Values After Handling:")
print(statistics_df.isnull().sum())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Filter rows for 'aces' statistics
aces_data = statistics_df[statistics_df['statistic_name'] == 'aces']

# Calculate total aces per match (home + away)
aces_data['total_aces'] = aces_data['home_value'] + aces_data['away_value']

# Plot histograms to visualize the distribution
plt.figure(figsize=(18, 6))

# Home Aces Histogram
plt.subplot(1, 3, 1)
sns.histplot(aces_data['home_value'], bins=20, kde=True, color='blue', alpha=0.7, edgecolor='gray')
plt.title("Distribution of Home Aces", fontsize=14)
plt.xlabel("Home Aces", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(False)

# Away Aces Histogram
plt.subplot(1, 3, 2)
sns.histplot(aces_data['away_value'], bins=20, kde=True, color='green', alpha=0.7, edgecolor='gray')
plt.title("Distribution of Away Aces", fontsize=14)
plt.xlabel("Away Aces", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(False)

# Total Aces Histogram
plt.subplot(1, 3, 3)
sns.histplot(aces_data['total_aces'], bins=20, kde=True, color='purple', alpha=0.7, edgecolor='gray')
plt.title("Distribution of Total Aces per Match", fontsize=14)
plt.xlabel("Total Aces", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(False)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# %%
# Filter rows corresponding to 'aces'
aces_data = statistics_df.query("statistic_name == 'aces'")

# Calculate total aces for home and away teams
total_aces_home = aces_data['home_value'].sum()
total_aces_away = aces_data['away_value'].sum()

# Aggregate total aces across all matches
total_aces = total_aces_home + total_aces_away

# Determine the number of unique matches
num_matches = aces_data['match_id'].nunique()

# Calculate the average number of aces per match
average_aces = total_aces / num_matches if num_matches > 0 else 0

# Print the result
print(f"Average number of aces per match: {average_aces:.2f}")


# %%
aces_data

# %%
import matplotlib.pyplot as plt

# Filter rows for 'aces' statistics
aces_data = statistics_df[statistics_df['statistic_name'] == 'aces']

# Calculate total aces per match (home + away)
aces_data['total_aces'] = aces_data['home_value'] + aces_data['away_value']

# Plot histogram of total aces
plt.figure(figsize=(10, 6))
plt.hist(aces_data['total_aces'], bins=20, color='skyblue', edgecolor='gray', alpha=0.7)  # رنگ حاشیه binها به خاکستری
plt.title("Distribution of Total Aces per Match", fontsize=16, color="gray", loc="left")
plt.xlabel("Total Aces", fontsize=12, color="gray", loc="left")
plt.ylabel("Frequency", fontsize=12, color="gray", loc="bottom")

# حذف خطوط شبکه (grid)
plt.grid(False)  # غیر فعال کردن خطوط grid

# حذف حاشیه‌های راست و بالا
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# تغییر رنگ خطوط محور سمت چپ و پایین به خاکستری
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# تغییر رنگ اعداد روی محور‌ها به خاکستری
plt.tick_params(axis='x', colors='gray')  # رنگ اعداد محور X
plt.tick_params(axis='y', colors='gray')  # رنگ اعداد محور Y

# نمایش نمودار
plt.tight_layout()
plt.show()

# Calculate average number of aces per match
total_aces = aces_data['total_aces'].sum()
num_matches = aces_data['match_id'].nunique()
average_aces_per_match = total_aces / num_matches if num_matches > 0 else 0

print(f"Average aces per match: {average_aces_per_match:.2f}")


# %%
import seaborn as sns

# نمونه داده‌ها (فرضی)
data = {'Home Aces': [total_aces_home], 'Away Aces': [total_aces_away]}
df = pd.DataFrame(data)

# رسم Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Aces: Home vs Away', fontsize=16, color='gray')
# تغییر رنگ خطوط محور
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# تغییر رنگ اعداد روی محورها
plt.tick_params(axis='x', colors='gray')  # رنگ اعداد محور X
plt.tick_params(axis='y', colors='gray')  # رنگ اعداد محور Y
plt.show()

# %%
# داده‌های مربوط به آیس‌ها
labels = ['Aces']
home_aces = [total_aces_home]
away_aces = [total_aces_away]

# تنظیمات اندازه نمودار
plt.figure(figsize=(10, 8))

# رسم نمودار میله‌ای
bar_width = 0.4
index = np.arange(len(labels))

# رسم میله‌ها برای خانه و مهمان
plt.bar(index - bar_width/2, home_aces, bar_width, label='Home Aces', color='#09122C')
plt.bar(index + bar_width/2, away_aces, bar_width, label='Away Aces', color='#E17564')

# عنوان و برچسب‌ها
plt.title('Total Aces: Home vs Away', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.xlabel('Teams', fontsize=10, color='gray', loc='left')
plt.ylabel('Number of Aces', fontsize=10, color='gray', loc='bottom')

# تنظیمات نمودار
plt.xticks(index, labels, fontsize=12, color='gray')
plt.yticks(fontsize=12, color='gray')  # تغییر رنگ اعداد محور Y به خاکستری
plt.legend(title="Teams", loc="best", labelcolor='gray')

# تغییر رنگ خطوط محور
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')
# تعریف افسانه
legend = plt.legend(title="Teams", loc="best", labelcolor='gray', frameon=False)

# تغییر متن و رنگ عنوان افسانه
legend.set_title("Teams")
legend.get_title().set_color("gray")  # تغییر رنگ عنوان

# تغییر رنگ اعداد روی محورها
plt.tick_params(axis='x', colors='gray')  # رنگ اعداد محور X
plt.tick_params(axis='y', colors='gray')  # رنگ اعداد محور Y
# تغییر رنگ لیبل‌های افسانه
for text in legend.get_texts():
    text.set_color("gray")  # تغییر رنگ متن‌های افسانه به خاکستری

# نمایش نمودار
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# رسم نمودار دایره‌ای با رنگ‌های متنوع‌تر
plt.figure(figsize=(6, 6))
colors = ['#09122C', '#E17564', '#C4D9FF']  # رنگ‌های متنوع
labels = ['Home Aces', 'Away Aces', 'Other Aces']  # اضافه کردن دسته "Other" به عنوان مثال

values = [total_aces_home, total_aces_away, 100]  # 100 مقدار تخمینی برای "Other"

# رسم نمودار دایره‌ای
wedges, _, autotexts = plt.pie(values, autopct='%1.1f%%', colors=colors, startangle=90)

# افزودن legend با تغییر رنگ متن به خاکستری
legend = plt.legend(wedges, labels, title="Teams", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, labelcolor='gray', title_fontsize=12)

# تغییر رنگ عنوان (Teams)
legend.get_title().set_color('gray')

# تغییر استایل متن‌های درصد
for autotext in autotexts:
    autotext.set_color('gray')
    autotext.set_fontsize(10)

plt.title('Percentage of Aces: Home vs Away vs Other', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.tight_layout()
plt.show()


# %% [markdown]
# # **8. Is there a difference in the number of double faults based on gender?What is the average number of aces per match?**

# %%
statistics_df.head(10)

# %%
statistics_df.info()

# %%
# !pip install scipy

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Merge dataframes based on match_id and filter double faults
home_double_faults = statistics_df[statistics_df['statistic_name'] == 'double_faults'].merge(home_team_df, on='match_id')
away_double_faults = statistics_df[statistics_df['statistic_name'] == 'double_faults'].merge(away_team_df, on='match_id')

# Group data by gender to get the total double faults for home and away teams
total_home_faults = home_double_faults.groupby('gender')['home_value'].sum().reset_index()
total_away_faults = away_double_faults.groupby('gender')['away_value'].sum().reset_index()

# Merge the home and away totals based on gender
total_faults = pd.merge(total_home_faults, total_away_faults, on='gender', how='outer')

# Melt data for plotting
total_faults_melted = pd.melt(total_faults, id_vars='gender', value_vars=['home_value', 'away_value'], var_name='team_type', value_name='double_faults')

# Display the aggregated data
print(total_faults)

# Colors for the barplot
colors = ['#09122C', '#E17564']  # Colors for home and away teams

# Plotting the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='double_faults', hue='team_type', data=total_faults_melted, palette=colors, dodge=True)

# Customizing the legend and labels
plt.legend(title="Teams", loc="best")
legend = plt.legend(fontsize=10, frameon=False, labelcolor='gray')
legend.get_texts()[0].set_text('Home')
legend.get_texts()[1].set_text('Away')
plt.title('Double Faults Based on Gender', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.xlabel('Gender', loc='left', color='gray', fontsize=10)
plt.ylabel('Double Faults Count', loc='bottom', color='gray', fontsize=10)

# Adjusting the tick parameters and removing borders
plt.tick_params(colors='gray')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

plt.show()

# Conducting the t-test to check for differences in double faults based on gender

# Extracting home and away double faults for male and female players
male_home_faults = home_double_faults[home_double_faults['gender'] == 'Male']['home_value']
female_home_faults = home_double_faults[home_double_faults['gender'] == 'Female']['home_value']
male_away_faults = away_double_faults[away_double_faults['gender'] == 'Male']['away_value']
female_away_faults = away_double_faults[away_double_faults['gender'] == 'Female']['away_value']

# T-test for home team
t_stat_home, p_val_home = ttest_ind(male_home_faults.dropna(), female_home_faults.dropna())

# T-test for away team
t_stat_away, p_val_away = ttest_ind(male_away_faults.dropna(), female_away_faults.dropna())

# Print the results of the t-tests
print(f"T-test (Home): t-statistic = {t_stat_home}, p-value = {p_val_home}")
print(f"T-test (Away): t-statistic = {t_stat_away}, p-value = {p_val_away}")

# Interpretation of the p-values:
if p_val_home < 0.05:
    print("There is a significant difference in double faults for home teams based on gender.")
else:
    print("There is no significant difference in double faults for home teams based on gender.")

if p_val_away < 0.05:
    print("There is a significant difference in double faults for away teams based on gender.")
else:
    print("There is no significant difference in double faults for away teams based on gender.")


# %% [markdown]
# *clean 'home_team_df' and 'away team_df' then rerun the plot code*

# %%
home_team_df['height'].fillna(home_team_df['height'].mean(), inplace=True)
home_team_df['weight'].fillna(home_team_df['weight'].mean(), inplace=True)

home_team_df.dropna(subset=['current_prize', 'total_prize', 'country', 'current_rank'], inplace=True)

home_team_df['current_prize'] = home_team_df['current_prize'].astype(float)
home_team_df['total_prize'] = home_team_df['total_prize'].astype(float)

home_team_df['gender'] = home_team_df['gender'].astype('category')
home_team_df['country'] = home_team_df['country'].astype('category')

home_team_df.drop_duplicates(subset='match_id', inplace=True)


home_team_df.drop(columns=['turned_pro', 'residence', 'birthplace'], inplace=True, errors='ignore')

away_team_df['height'].fillna(away_team_df['height'].mean(), inplace=True)
away_team_df['weight'].fillna(away_team_df['weight'].mean(), inplace=True)

away_team_df.dropna(subset=['current_prize', 'total_prize', 'country', 'current_rank'], inplace=True)

away_team_df['current_prize'] = away_team_df['current_prize'].astype(float)
away_team_df['total_prize'] = away_team_df['total_prize'].astype(float)

away_team_df['gender'] = away_team_df['gender'].astype('category')
away_team_df['country'] = away_team_df['country'].astype('category')

away_team_df.drop_duplicates(subset='match_id', inplace=True)


away_team_df.drop(columns=['turned_pro', 'residence', 'birthplace'], inplace=True, errors='ignore')

# %%
# Filter double fault datas
home_double_faults = statistics_df[statistics_df['statistic_name'] == 'double_faults'].merge(home_team_df, on='match_id')
away_double_faults = statistics_df[statistics_df['statistic_name'] == 'double_faults'].merge(away_team_df, on='match_id')

# Total double fault for every gender
total_home_faults = home_double_faults.groupby('gender')['home_value'].sum().reset_index()
total_away_faults = away_double_faults.groupby('gender')['away_value'].sum().reset_index()

# Merge totals
total_faults = pd.merge(total_home_faults, total_away_faults, on='gender', how='outer')

# Convert types of data to plot 
total_faults_melted = pd.melt(total_faults, id_vars='gender', value_vars=['home_value', 'away_value'], var_name='team_type', value_name='double_faults')

# Display datas
print(total_faults)

# Colors
colors = ['#09122C', '#E17564']  # Colors for home and away teams
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='double_faults', hue='team_type', data=total_faults_melted, palette=colors, dodge=True)

# Set legend
plt.legend(title="Teams", loc="best")
legend = plt.legend(fontsize=10, frameon=False,labelcolor='gray')
legend.get_texts()[0].set_text('Home')
legend.get_texts()[1].set_text('Away')
plt.title('Double Faults Based on Gender', loc='left', color='gray', fontweight='bold')
plt.xlabel('Gender', loc='left', color='gray', fontsize=10)
plt.ylabel('Double Faults Count', loc='bottom', color='gray', fontsize=10)

# Change lines and colors to gray
plt.tick_params(colors='gray')

# Delete right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

plt.show()


# %% [markdown]
# # **Impact of Gender and Team Type on Double Faults**
# **It can be concluded from the data that gender and team type affect the number of double faults.**
# **Double faults are more frequent among women, and for men, the difference between home and away teams is more pronounced.**
# 

# %% [markdown]
# # **9. Which player has won the most tournaments in a single month?**

# %%
tournament_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/tournament"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/tournament/" + file)
    tournament_df = pd.concat([tournament_df, single_stats], axis= 'rows', ignore_index=True)


# %%
tournament_df

# %%
tournament_df.info()

# %% [markdown]
# ## **Missing values**

# %%
tournament_df.isnull().sum()

# %% [markdown]
# ## **Handle Missing values**

# %%
tournament_df = tournament_df.drop('tournament_unique_id', axis=1)

# %% [markdown]
# ## **Fill missing categorical columns with mode**

# %% [markdown]
# برای ستون‌های متنی، از مد استفاده کردیم چون منطقی‌ترین جایگزین برای داده‌های متنی است

# %%
mode_value = tournament_df['tennis_points'].mode()[0]  
tournament_df['tennis_points'].fillna(mode_value, inplace=True)  
mode_value_g = tournament_df['ground_type'].mode()[0]  
tournament_df['ground_type'].fillna(mode_value_g, inplace=True)  


# %% [markdown]
# # **Check for Duplicates**

# %%
total_duplicates = tournament_df.duplicated().sum()
print(f'Total number of duplicates before removal: {total_duplicates}')


# %% [markdown]
# # **Check for Missing Values After Handling**

# %%
print("Missing Values After Handling:")
print(tournament_df.isnull().sum())

# %%
# Step1: combine winner information

home_wins = event_df[event_df['winner_code'] == 1].merge(home_team_df, on='match_id')

away_wins = event_df[event_df['winner_code'] == 2].merge(away_team_df, on='match_id')

all_wins = pd.concat([home_wins[['match_id', 'player_id']], away_wins[['match_id', 'player_id']]])

# Step2: Merge winners and tournament information
all_wins = all_wins.merge(tournament_df[['match_id', 'tournament_id', 'tournament_name']], on='match_id')

# Step3: Number of tournament of each player win
tournament_wins = all_wins.groupby('player_id')['tournament_id'].nunique().reset_index(name='tournament_count')

# Step4: Max player win tournamnet
top_player = tournament_wins.loc[tournament_wins['tournament_count'].idxmax()]

top_player_id = top_player['player_id']
top_player_wins = top_player['tournament_count']

# Step5: Find name of player from player_id
if top_player_id in home_team_df['player_id'].values:
    top_player_name = home_team_df[home_team_df['player_id'] == top_player_id]['name'].values[0]
elif top_player_id in away_team_df['player_id'].values:
    top_player_name = away_team_df[away_team_df['player_id'] == top_player_id]['name'].values[0]
else:
    top_player_name = "No Player Found"

print(f"Player with most tournament wins: {top_player_name} (ID: {top_player_id})")
print(f"Number of tournament wins: {top_player_wins}")

# %% [markdown]
# # **10. Is there a correlation between a player's height and their ranking?**

# %%
import matplotlib.patches as patches

# Step 1: Extract player information from home and away teams
home_players = home_team_df[['player_id', 'height', 'current_rank']]
away_players = away_team_df[['player_id', 'height', 'current_rank']]

# Merge home and away teams
all_players = pd.concat([home_players, away_players])

# Remove rows with NaN in the current_rank column
all_players = all_players.dropna(subset=['current_rank'])

# Remove duplicate rows based on player_id
all_players = all_players.drop_duplicates(subset='player_id')

# Step 2: Calculate Pearson correlation between height and ranking
correlation = all_players['height'].corr(all_players['current_rank'])
print(f'Correlation between height and current ranking: {correlation:.2f}')

# Step 3: Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_players, x='current_rank', y='height', palette='black', edgecolor='w', s=100, alpha=0.7)
plt.title('Scatter Plot of Player Height Vs. Current Ranking', fontsize=14, fontweight='bold', loc='left', color='gray')
plt.xlabel('Player Current Rank', loc='left', color='gray')
plt.ylabel('Player Height (m)', loc='bottom', color='gray')

# Change the axis tick colors to gray
plt.gca().tick_params(axis='x', colors='gray')
plt.gca().tick_params(axis='y', colors='gray')

# Change the axis spine color to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# Remove right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Add a fluorescent rectangle for players ranked 1 to 50
plt.gca().add_patch(patches.Rectangle((1, all_players['height'].min()), 
                                       50, all_players['height'].max() - all_players['height'].min(), 
                                       linewidth=2, edgecolor='none', facecolor='red', alpha=0.3))

# Add label to Legend
plt.legend(title='Top 50 Players', loc='upper right', fontsize='small', title_fontsize='medium', labelcolor='gray', frameon=False)
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(patches.Rectangle((0,0), 1, 1, color='red', alpha=0.3))
labels.append('Top 50 Players')
plt.legend(handles=handles, labels=labels, labelcolor='gray', frameon=False)

plt.show()


# %% [markdown]
# ### **The correlation between a player's height and their current ranking is 0.03. This value suggests a very weak positive correlation, meaning that there is almost no relationship between the player's height and their ranking. In other words, height does not significantly influence a player's ranking.**

# %% [markdown]
# # **11. What is the average duration of matches?**

# %%
# Calculate the total duration (total match time) for each match
time_df['total_duration'] = (
    time_df['period_1'] +
    time_df['period_2'] +
    time_df['period_3'] +
    time_df['period_4'] +
    time_df['period_5']
)

# 1. Remove matches where the total duration is zero
valid_durations = time_df[time_df['total_duration'] > 0]['total_duration']

# 2. Calculate the average duration
average_duration = valid_durations.mean() / 60  # Convert to minutes

print(f"Average duration of a match: {average_duration:.2f} min")


# %% [markdown]
# The **average duration** of a match is calculated based on the total playing time across multiple matches. After analyzing the data, it was found that the **average match duration is 100.40 minutes**. 
# 
# This number reflects the combined length of all periods in the match (such as periods 1 to 5). The reason for this duration might be influenced by various factors including the type of sport, number of overtime periods, breaks, and any additional stoppage time. The figure of 100.40 minutes provides an overall estimation of how long players typically engage in the game, considering these variables.
# 

# %% [markdown]
# # **12. What is the average number of games per set in men's matches compared to women's matches?**

# %%
p_by_p_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_point_by_point_parquet"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_point_by_point_parquet/" + file)
    p_by_p_df = pd.concat([p_by_p_df, single_stats], axis= 'rows', ignore_index=True)


# %%
p_by_p_df

# %%
p_by_p_df.info()

# %% [markdown]
# ## **Missing values**

# %%
p_by_p_df.isnull().sum()

# %%
# Merge the p_by_p dataframe with the home and away team information based on match_id
merged_home = p_by_p_df.merge(home_team_df[['match_id', 'gender']], on='match_id', how='left', suffixes=('', '_home'))
merged_data = merged_home.merge(away_team_df[['match_id', 'gender']], on='match_id', how='left', suffixes=('', '_away'))

# Combine gender information (manage potential null values)
merged_data['gender'] = merged_data['gender'].fillna(merged_data['gender_away'])

# Drop the 'gender_away' column
merged_data = merged_data.drop(['gender_away'], axis=1)

merged_data


# %%
# Group by match_id, set_id, and gender to count unique games in each set
games_per_set = merged_data.groupby(['match_id', 'set_id', 'gender'])['game_id'].nunique().reset_index()
print("Games per set:\n", games_per_set.head())

# Calculate the average number of games per set for each gender
average_games = games_per_set.groupby('gender')['game_id'].mean().reset_index()
average_games.columns = ['Gender', 'Average_Games_Per_Set']
print("Average games per set by gender:\n", average_games)


# %%
# Check for match_ids present in p_by_p_df but missing in home_team_df
missing_in_home = p_by_p_df[~p_by_p_df['match_id'].isin(home_team_df['match_id'])]

# Check for match_ids present in p_by_p_df but missing in away_team_df
missing_in_away = p_by_p_df[~p_by_p_df['match_id'].isin(away_team_df['match_id'])]

print("Missing in home team dataframe:\n", missing_in_home['match_id'].unique())
print("Missing in away team dataframe:\n", missing_in_away['match_id'].unique())


# %%
# Create a bar plot to display the average number of games per set by gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Average_Games_Per_Set', data=average_games, palette='pastel')

# Add titles and labels
plt.title('Average Number of Games Per Set by Gender', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.xlabel('Gender', fontsize=12, color='gray', loc='left')
plt.ylabel('Average Games Per Set', fontsize=12, color='gray', loc='bottom')

# Remove top and right spines for a cleaner look
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# Set tick parameters to display in gray color
plt.tick_params(colors='gray')

# Display the plot
plt.show()


# %% [markdown]
# ### Based on the provided chart, several conclusions can be drawn:
# 
# - **Similarity in Match Play:** 
# The average number of games per set for both genders (men and women) is quite close. This indicates a balance in the level of competition in both categories of matches.
# 
# - **Slightly Higher for Men:**
# With averages of 8.60 games for men and 8.49 for women, it can be concluded that, on average, men’s matches involve slightly more games per set. This could be related to factors such as playing styles, serve power, or overall strategies employed by male players.
# 
# - **Similar Competitiveness and Challenges:**
# Although the number of games in men’s matches is slightly higher, this difference is not significant and suggests that both categories offer similarly competitive and challenging environments.
# 
# - **Analysis of Playing Style:** 
# These results may indicate strategic and stylistic differences between men’s and women’s matches, which could influence coaching decisions and video analyses of games.
# 
# In conclusion, both genders exhibit closely matched competition and a high level of skill, even if there is a small variance in the average number of games per set. 

# %% [markdown]
# # **13. What is the distribution of left-handed versus right-handed players?**

# %%
home_team_df

# %%
away_team_df

# %%
home_team_df['plays'] = home_team_df['plays'].str.title()
away_team_df['plays'] = away_team_df['plays'].str.title()

distribution_home = home_team_df['plays'].value_counts()
distribution_away = away_team_df['plays'].value_counts()

home_plays = home_team_df[['plays']].copy()
away_plays = away_team_df[['plays']].copy()

all_plays = pd.concat([home_plays, away_plays])

# شمارش تعداد بازیکنان راست‌دست و چپ‌دست
distribution_all = all_plays['plays'].value_counts()

print("Distribution of Left-Handed vs. Right-Handed Players:")
print(distribution_all)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Colors
# colors = ['#E1C9F0', '#AAD6F0']  # Colors for home and away teams
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=distribution_all.index, y=distribution_all.values, data=total_faults_melted, palette='pastel', dodge=True)


# افزودن عدد دقیق روی هر ستون
for index, value in enumerate(distribution_all.values):
    plt.text(index, value + 1, str(value), color='gray', ha="center", fontsize=12)


plt.title('Distribution of Left-Handed Versus Right-Handed Players', loc='left', color='gray' ,fontweight='bold')
plt.xlabel('Hand Preference', loc='left', color='gray', fontsize=10)
plt.ylabel('Player Count', loc='bottom', color='gray', fontsize=10)

# Change lines and colors to gray
plt.tick_params(colors='gray')

# Delete right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

plt.show()


# رسم نمودار دایره‌ای
plt.figure(figsize=(10, 8))
labels = ['Left_Handed', 'Right_Handed']
colors = sns.color_palette("pastel", 2)  

wedges, texts, autotexts = plt.pie(distribution_all.values, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Percentage of Left-handed vs Right-handed Players', fontsize=16, color='gray', loc='left', fontweight='bold')

# اضافه کردن legend
plt.legend(wedges, ['Right-handed', 'Left-handed'], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, labelcolor='gray')
legend.get_title().set_color('gray')
for autotext in autotexts:
    autotext.set_color('gray')
    autotext.set_fontsize(10)

# تنظیم رنگ و فونت درصدها
plt.setp(autotexts, color="gray", fontweight="bold")
plt.title('Percentage of Left-Handed Versus Right-Handed Players', fontsize=16, color='gray', loc='left', fontweight='bold')
# # تغییر متن و رنگ عنوان افسانه
# legend.set_title("Player Type")
# legend.get_title().set_color("gray")  # تغییر رنگ عنوان
# تغییر رنگ لیبل‌های افسانه
for text in legend.get_texts():
    text.set_color("gray")  # تغییر رنگ متن‌های افسانه به خاکستری

# تغییر رنگ اعداد روی محورها
plt.tick_params(axis='x', colors='gray')  # رنگ اعداد محور X
plt.tick_params(axis='y', colors='gray')  # رنگ اعداد محور Y

plt.show()






# %% [markdown]
# # Analysis of Bar Chart  
# 
# This bar chart illustrates the distribution of left-handed versus right-handed players.  
# 
# ### Chart Title:  
# - **"Distribution of Left-Handed Versus Right-Handed Players"** clearly indicates a comparison between the two groups.  
# 
# ### Axes:  
# - **Horizontal Axis (Hand Preference)**: Displays two categories — "Right-Handed" and "Left-Handed."  
# - **Vertical Axis (Player Count)**: Represents the number of players, ranging from 0 to 350.  
# 
# ### Ratios:  
# - **Right-Handed Players**: 312, showing a significant prevalence.  
# - **Left-Handed Players**: Only 46, highlighting a low representation compared to right-handed players.  
# 
# ### Overall Analysis:  
# - The chart reveals a notable predominance of right-handed players, potentially influenced by various factors such as abilities, social norms, and educational influences.  
# 
# ### Conclusion:  
# - Most players in this sample are right-handed, which could inform coaches and analysts in their selection and training strategies.  
# 
# ---  
# 
# # Analysis of Pie Chart  
# 
# This pie chart depicts the percentage distribution of left-handed and right-handed players.  
# 
# ### Chart Title:  
# - **"Percentage of Left-Handed Versus Right-Handed Players"** emphasizes the comparison of player ratios.  
# 
# ### Ratios:  
# - **Right-Handed Players**: 87.2%, demonstrating dominance in the sample.  
# - **Left-Handed Players**: Only 12.8%, indicating a significant minority.  
# 
# ### Overall Analysis:  
# - The chart confirms that most players are right-handed, possibly due to genetic, cultural, or historical factors. Left-handed players are often at a disadvantage in a predominantly right-handed society.  
# 
# ### Conclusion:  
# - The findings offer insights for coaches and analysts regarding player composition, suggesting tailored training approaches for left-handed players.  

# %% [markdown]
# # **14. What is the most common type of surface used in tournaments?**

# %%
# tournament_df = pd.DataFrame()
# for file in os.listdir("tennis_data_20231212/raw/raw_match_parquet/tournament"):
#     single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_match_parquet/tournament/" + file)
#     tournament_df = pd.concat([tournament_df, single_stats], axis= 'rows', ignore_index=True)

# %%
tournament_df

# %%
tournament_df.info()

# %%
tournament_df_cleaned = tournament_df.dropna(subset=['ground_type'])

# گروه‌بندی بر اساس ground_type و شمارش تعداد هر نوع سطح
surface_count = tournament_df_cleaned['ground_type'].value_counts()

# نمایش رایج‌ترین نوع سطح
most_common_surface = surface_count.idxmax()
count_most_common_surface = surface_count.max()

print(f"The most common type of surface is: {most_common_surface} with {count_most_common_surface} tournaments.")


# %%
unique_ground_types = tournament_df_cleaned['ground_type'].unique()  
number_of_ground_types = len(unique_ground_types)  
print(unique_ground_types)  
print(number_of_ground_types)


# %%
ground_type_counts = tournament_df_cleaned['ground_type'].value_counts()  
print(ground_type_counts)

# %%
# import matplotlib.pyplot as plt  
# import numpy as np  

# # تعداد واقعی هر نوع سطح از داده‌ها
# counts = np.array([323, 353, 84, 3, 1])  

# # محاسبه درصدها
# percentages = (counts / counts.sum()) * 100  

# # برچسب‌ها و رنگ‌ها
# labels = ['Red clay', 'Hardcourt outdoor', 'Hardcourt indoor', 'Carpet indoor', 'Synthetic outdoor']  
# colors = ['#d09c9c', '#3A3960', '#bdcff2', '#e5d673', '#bdf2c2']  

# # رسم نمودار دایره‌ای با درصدهای درست
# plt.figure(figsize=(10, 8))  
# wedges, texts, autotexts = plt.pie(counts, autopct=lambda p: f'{p:.1f}%', startangle=90, colors=colors)  

# plt.title('Percentage of Common Type of Surface Used in Tournaments', fontsize=16, color='gray', loc='left', fontweight='bold')  

# # اضافه کردن راهنما
# legend = plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, labelcolor='gray')  
# legend.get_title().set_color('gray')  

# # تنظیم رنگ و اندازه فونت درصدها
# for autotext in autotexts:  
#     autotext.set_color('gray')  
#     autotext.set_fontsize(10)  

# plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np  

# داده‌ها  
labels = ['Hardcourt outdoor','Red clay',  'Hardcourt indoor', 'Carpet indoor', 'Synthetic outdoor']
# counts = ground_type_counts
colors = ['#3A3960', '#d09c9c', '#bdcff2', '#e5d673', '#bdf2c2']  

# رسم نمودار ستونی  
plt.figure(figsize=(10, 6))  
plt.bar(labels, ground_type_counts, color=colors)  

# اضافه کردن توضیحات
plt.xlabel("Surface Type", loc='left', color='gray' ,fontsize=12)
plt.ylabel("Count", loc='bottom', color='gray', fontsize=12)
plt.title("Distribution of Surface Types in Tournaments",loc='left', color='gray', fontsize=14, fontweight="bold")


# Change lines and colors to gray
plt.tick_params(colors='gray')

# Delete right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')


# اضافه کردن مقدار بالای هر ستون
for i, v in enumerate(ground_type_counts):
    plt.text(i, v + 10, str(v), ha='center',color='gray', fontsize=12)

plt.show()


# %% [markdown]
# # **15. How many distinct countries are represented in the dataset?**

# %%
home_team_df

# %%
combined_df = pd.concat([home_team_df, away_team_df])

# %%
distinct_country = combined_df['country'].nunique()
print(f"There are {distinct_country} distinct countries represented in the dataset.")


# %% [markdown]
# # **16. Which player has the highest winning percentage against top 10 ranked opponents?**

# %%
# مرحله 1: استخراج اطلاعات بازیکنان و دراپ کردن مقادیر نال
# انتخاب ستون‌های نام، رتبه و جنسیت از جداول home_team و away_team
top_players_home = home_team_df[['name', 'current_rank', 'gender']].dropna(subset=['current_rank'])
top_players_away = away_team_df[['name', 'current_rank', 'gender']].dropna(subset=['current_rank'])

# ترکیب دو لیست بازیکنان و حذف مقادیر تکراری
top_players = pd.concat([top_players_home, top_players_away]).drop_duplicates()

# مرحله 2: جدا کردن بازیکنان بر اساس جنسیت
# 10 بازیکن برتر مرد و 10 بازیکن برتر زن
top_10_men = top_players[top_players['gender'] == 'M'].sort_values(by='current_rank').head(10)
top_10_women = top_players[top_players['gender'] == 'F'].sort_values(by='current_rank').head(10)

# تبدیل نام بازیکنان به لیست برای استفاده در مراحل بعدی
top_10_men_names = top_10_men['name'].tolist()
top_10_women_names = top_10_women['name'].tolist()

# %%
top_10_men

# %%
top_10_women

# %%
# مرحله 2: ادغام جداول برای دسترسی به اطلاعات تیم‌های خانگی و میهمان
home_matches = event_df.merge(home_team_df[['match_id', 'name']], on='match_id', how='left')
away_matches = event_df.merge(away_team_df[['match_id', 'name']], on='match_id', how='left')

# تغییر نام ستون‌ها برای تمایز بین بازیکن خانگی و میهمان
home_matches = home_matches.rename(columns={'name': 'home_player'})
away_matches = away_matches.rename(columns={'name': 'away_player'})

# مرحله 3: ادغام اطلاعات بازیکنان خانگی و میهمان در یک جدول
all_matches = home_matches.merge(away_matches[['match_id', 'away_player']], on='match_id', how='left')

# مرحله 4: فیلتر کردن مسابقات که یکی از بازیکنان 10 بازیکن برتر باشد
top_10_matches = all_matches[(all_matches['home_player'].isin(top_10_men_names + top_10_women_names)) |
                             (all_matches['away_player'].isin(top_10_men_names + top_10_women_names))]

# مرحله 5: محاسبه برد و باخت
# تابعی برای تعیین برنده و بازنده
def determine_win_loss(row):
    if row['winner_code'] == 1:  # تیم خانگی برنده است
        winner = row['home_player']
        loser = row['away_player']
    else:  # تیم میهمان برنده است
        winner = row['away_player']
        loser = row['home_player']
    return winner, loser

# اعمال تابع برای تعیین برنده و بازنده
top_10_matches['winner'], top_10_matches['loser'] = zip(*top_10_matches.apply(determine_win_loss, axis=1))

# مرحله 6: فیلتر کردن بازیکنانی که با 10 بازیکن برتر بازی کرده‌اند اما جزو 10 نفر نیستند
non_top_10_matches = top_10_matches[(~top_10_matches['winner'].isin(top_10_men_names + top_10_women_names)) |
                                    (~top_10_matches['loser'].isin(top_10_men_names + top_10_women_names))]

# مرحله 7: محاسبه تعداد بردها و باخت‌ها برای هر بازیکن مقابل 10 بازیکن برتر
# شمارش تعداد بردها
win_counts = non_top_10_matches['winner'].value_counts().reset_index()
win_counts.columns = ['player', 'wins']

# شمارش تعداد باخت‌ها
loss_counts = non_top_10_matches['loser'].value_counts().reset_index()
loss_counts.columns = ['player', 'losses']

# مرحله 8: ادغام برد و باخت‌ها در یک جدول نهایی
player_stats = pd.merge(win_counts, loss_counts, on='player', how='outer').fillna(0)

# نمایش جدول نهایی
player_stats['wins'] = player_stats['wins'].astype(int)
player_stats['losses'] = player_stats['losses'].astype(int)
player_stats = player_stats.sort_values(by=['wins', 'losses'], ascending=False)

player_stats


# %% [markdown]
# # **17. What is the average number of breaks of serve per match?**

# %%
# Step 1: Filter the statistics for 'service_games_played' and 'break_points_saved'
service_games_df = statistics_df[statistics_df['statistic_name'] == 'service_games_played']
break_points_saved_df = statistics_df[statistics_df['statistic_name'] == 'break_points_saved']

# Step 2: Merge the two DataFrames to compare service games and break points saved
merged_df = service_games_df.merge(break_points_saved_df, on='match_id', suffixes=('_service_games', '_break_points_saved'))

# Step 3: Calculate breaks of serve
merged_df['home_breaks_of_serve'] = merged_df['home_value_service_games'] - merged_df['home_value_break_points_saved']
merged_df['away_breaks_of_serve'] = merged_df['away_value_service_games'] - merged_df['away_value_break_points_saved']

# Step 4: Calculate the total number of breaks per match (home + away)
merged_df['total_breaks_of_serve'] = merged_df['home_breaks_of_serve'] + merged_df['away_breaks_of_serve']

# Step 5: Calculate the average number of breaks of serve per match
average_breaks_per_match = merged_df['total_breaks_of_serve'].mean()

# Display the result
print(f"The average number of breaks of serve per match is {average_breaks_per_match:.2f}.")


# %%
# گام 1: محاسبه مجموع بریک‌های خانگی و مهمان
total_home_breaks = merged_df['home_breaks_of_serve'].sum()
total_away_breaks = merged_df['away_breaks_of_serve'].sum()

# محاسبه درصد بریک‌های خانگی و مهمان
total_breaks = total_home_breaks + total_away_breaks
home_breaks_percentage = (total_home_breaks / total_breaks) * 100
away_breaks_percentage = (total_away_breaks / total_breaks) * 100


# %%
print(f"Total home breaks: {total_home_breaks}")
print(f"Total away breaks: {total_away_breaks}")
print(f"Total breaks: {total_breaks}")


# %%
average_home_breaks = merged_df['home_breaks_of_serve'].mean()
average_away_breaks = merged_df['away_breaks_of_serve'].mean()

# Colors
colors = ['#09122C', '#E17564']  # Colors for home and away teams
# Plot
plt.figure(figsize=(10, 6))

# نمودار میله‌ای بر اساس میانگین‌ها
plt.bar(['Home Breaks', 'Away Breaks'], [average_home_breaks, average_away_breaks], color=colors)
plt.title('Average Breaks of Serve per Match: Home vs Away', fontsize=16, color='gray', loc='left', fontweight='bold')
plt.ylabel('Average Breaks', fontsize=12, color='gray', loc='bottom')
plt.tick_params(colors='gray')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# Change lines and colors to gray
plt.tick_params(colors='gray')

# Delete right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Change the color of the left and bottom spines to gray
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

plt.show()


# %% [markdown]
# # **18. Number of player by countries?**

# %%
country_counts = combined_df['country'].value_counts()
country_counts

# %%
colors = ['#B5C0D0', '#CCD3CA', '#F5E8DD', '#EED3D9', '#F5DAD2', '#FCFFE0', '#BACD92', '#F1EAFF', '#D2E0FB', '#FEF9D9']

# تعداد بازیکنان بر اساس کشور
country_counts = combined_df['country'].value_counts()

country_counts.head(10).plot(kind='pie', figsize=(8, 8), 
                            #  colormap="Set3",  # استفاده از پالت پیش‌ساخته
                             colors=colors,  # تغییر رنگ‌های بخش‌ها
                             labels=[f'{label} ({count})' for label, count in zip(country_counts.head(10).index, country_counts.head(10))], 
                             autopct='%1.0f%%', 
                             textprops={'color': 'gray'})  # تغییر رنگ نوشته‌ها به خاکستری

plt.title("Top 10 Countries by Number of Tennis Players", color="gray")
plt.ylabel("")  # حذف برچسب پیش‌فرض
plt.show()


# %% [markdown]
# # **19. Average of player's weight?**

# %%
combined_df['weight'].mean()


# %% [markdown]
# # **20. Distribution of votes?**

# %% [markdown]
# ### **Load data**

# %%
votes_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_votes_parquet"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_votes_parquet/" + file)
    votes_df = pd.concat([votes_df, single_stats], axis='rows', ignore_index=True)

# %%
votes_df

# %%
votes_df.info()

# %% [markdown]
# ### **تحلیل‌های آماری** 

# %%
votes_df[['home_vote', 'away_vote']].describe()

# %% [markdown]
# ### **مقایسه تیم‌ها**
# ***بررسی تعداد رأی‌های بیشتر بین تیم میزبان و میهمان***

# %%
votes_df['winner'] = votes_df.apply(lambda row: 'home' if row['home_vote'] > row['away_vote'] else 'away', axis=1)
votes_df['winner'].value_counts()

# %% [markdown]
# ## Votes distribution

# %%
total_votes = votes_df[['home_vote', 'away_vote']].sum()
print(total_votes)

# %%
import matplotlib.pyplot as plt

# Colors
colors = ['#09122C', '#E17564']  # Colors for home and away teams

# رسم نمودار
plt.figure(figsize=(10, 5))
plt.bar(['Home', 'Away'], total_votes, color=colors)

# تنظیمات نمودار
plt.title('Vote Distribution', loc='left', color='gray', fontweight='bold', fontsize=14)
plt.xlabel('Team', color='gray', fontsize=12, loc='left')
plt.ylabel('Total Votes', color='gray', fontsize=12, loc='bottom')

# تغییر رنگ خطوط و حذف خطوط اضافی
plt.tick_params(colors='gray')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# اضافه کردن مقدار بالای هر ستون
for i, v in enumerate(total_votes):
    plt.text(i, v + 10, str(v), ha='center',color='gray', fontsize=12)

plt.show()


# %% [markdown]
# # **21. Difference votes between home and away teams ?**

# %%
votes_df['vote_diff'] = votes_df['home_vote'] - votes_df['away_vote']
votes_df['vote_diff'].describe()

# %% [markdown]
# ## تحلیل مسابقاتی که اختلاف رأی زیادی دارند

# %%
high_diff_matches = votes_df[abs(votes_df['vote_diff']) > votes_df['vote_diff'].mean()]
high_diff_matches


# %% [markdown]
# ## شناسایی مسابقات کم‌رأی یا بی‌رأی

# %%
no_votes = votes_df[(votes_df['home_vote'] == 0) & (votes_df['away_vote'] == 0)]
no_votes


# %% [markdown]
# ## نمودار پراکندگی برای مقایسه رأی‌ها

# %%
votes_df.plot.scatter(x='home_vote', y='away_vote', alpha=0.5, title='Home vs Away Votes')


# %% [markdown]
# # **23. Total games and sets number?**

# %%
power_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_tennis_power_parquet"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_tennis_power_parquet/" + file)
    power_df = pd.concat([power_df, single_stats], axis='rows', ignore_index=True)

# %%
power_df

# %%
total_games = power_df['game_num'].nunique()
total_sets = power_df['set_num'].nunique()
print(f'Total games are {total_games} and total sets are {total_sets}.')

# %% [markdown]
# ### **درصد وقوع شکست سرویس در کل گیم‌ها**

# %%
break_percentage = power_df['break_occurred'].mean() * 100
break_percentage


# %% [markdown]
# 
# ### **نمایش تعداد شکست سرویس‌ها در هر ست**

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات شکل و اندازه نمودار
plt.figure(figsize=(10, 6))

# رسم نمودار countplot
sns.countplot(
    x='set_num', 
    hue='break_occurred', 
    data=power_df, 
    palette=['#09122C', '#E17564']  # رنگ‌ها برای True و False
)

# عنوان و توضیحات نمودار
plt.title('Break Occurrences per Set', loc='left', color='gray', fontweight='bold', fontsize=14)
plt.xlabel('Set Number', color='gray', fontsize=12, loc='left', fontweight='bold')
plt.ylabel('Count', color='gray', fontsize=12, loc='bottom', fontweight='bold')

# تنظیم رنگ و ظاهر خطوط و محورهای نمودار
plt.tick_params(colors='gray', labelsize=10)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')

# تنظیمات راهنما (Legend)
legend = plt.legend(
    title='Break Occurred',  # عنوان راهنما
    title_fontsize=12,       # اندازه فونت عنوان راهنما
    fontsize=10,             # اندازه فونت آیتم‌های راهنما
    frameon=False,           # حذف کادر دور راهنما
    loc='best',              # مکان بهترین جایگاه برای راهنما
    labelcolor='gray'        # رنگ متن آیتم‌ها
)

# تغییر رنگ عنوان (Teams)
legend.get_title().set_color('gray')

# تغییر استایل متن‌های درصد
for autotext in autotexts:
    autotext.set_color('gray')
    autotext.set_fontsize(10)

# تغییر نام True و False در راهنما
legend.get_texts()[0].set_text('False')  # نامگذاری آیتم False
legend.get_texts()[1].set_text('True')   # نامگذاری آیتم True

# نمایش نمودار
plt.show()


# %% [markdown]
# # **24.What ratio of the bets have been successful?**

# %% [markdown]
# ## **Load data**

# %%
odds_df = pd.DataFrame()
for file in os.listdir("tennis_data_20231212/raw/raw_odds_parquet"):
    single_stats = pd.read_parquet("tennis_data_20231212/raw/raw_odds_parquet/" + file)
    odds_df = pd.concat([odds_df, single_stats], axis='rows', ignore_index=True)

# %%
odds_df

# %%
odds_df.info()

# %%
# محاسبه تعداد کل شرط‌ها
total_bets = len(odds_df)

# محاسبه تعداد شرط‌های موفق
successful_bets = odds_df['winnig'].sum()

# محاسبه نسبت شرط‌های موفق
success_ratio = successful_bets / total_bets

# نمایش نتیجه
print(f"Total Bets: {total_bets}")
print(f"Successful Bets: {successful_bets}")
print(f"Success Ratio: {success_ratio:.2%}")



