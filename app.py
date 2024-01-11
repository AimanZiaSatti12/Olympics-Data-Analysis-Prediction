import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
import time
import pickle
import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/MA/OneDrive/Desktop/athlete_events.csv')
region_df = pd.read_csv('C:/Users/MA/OneDrive/Desktop/noc_regions.csv')

df=preprocessor.preprocess(df, region_df)
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://www.nicepng.com/png/detail/177-1776002_olympic-logo-with-sports-summer-olympic-games.png')
user_menu=st.sidebar.radio(
    'Select an Option',
    ('Medal Tally','Overall Analysis','Country-Wise Analysis','Athlete-Wise Analysis',"Medal-Wise Analysis","Medal Predictor","Match Prediction","Cluster Analysis","About")
)

if user_menu=="Medal Tally":

    st.sidebar.header("Medal Tally")
    years,country=helper.country_year_list(df)

    selected_year=st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally=helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in "+ str(selected_year)+" Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country+ "Overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country +" performance in "+str(selected_year)+" Olympics")
    st.table(medal_tally)

if user_menu == "Overall Analysis":
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['ID'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.markdown(
        '<h1 style="color: #3498db;">Top Statistics</h1>',
        unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time=helper.data_over_time(df,'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.markdown(
        '<h1 style="color: #3498db;">Participating Nations Over The Years</h1>',
        unsafe_allow_html=True)
    #st.title("Participating Nations Over The Years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.markdown(
        '<h1 style="color: #3498db;">Events Over The Years</h1>',
        unsafe_allow_html=True)
    #st.title("Events Over The Years")
    st.plotly_chart(fig)

    athletes_per_year = df[df['Season'] == 'Summer'].groupby('Year')['ID'].count().reset_index()

    athlete_over_time = helper.data_over_time(df, 'ID')
    fig = px.line(athlete_over_time, x="Edition", y="ID",labels={'ID': 'No of Athletes'}, title='Number of Athletes per Year ')
    st.markdown(
        '<h1 style="color: #3498db;">Athletes Over The Years</h1>',
        unsafe_allow_html=True)
    #st.title("Athletes Over The Years")
    st.plotly_chart(fig)
    st.markdown(
        '<h1 style="color: #3498db;">No. Of Event Over Time(Every Sport)</h1>',
        unsafe_allow_html=True)
   # st.title("No. Of Event Over Time(Every Sport)")
    fig,ax=plt.subplots(figsize=(20,20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax=sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)
    st.markdown(
        '<h1 style="color: #3498db;">Most Successful Athletes</h1>',
        unsafe_allow_html=True)
    #st.title("Most Successful Athletes")
    sport_list=df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport=st.selectbox('Select a Sport',sport_list)
    x=helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu=='Country-Wise Analysis':

    st.sidebar.title('Country-Wise Analysis')
    country_list=df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country=st.sidebar.selectbox('Select a Country',country_list)

    country_df=helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")

    st.markdown(
        '<h1 style="color: #006400;">Top 10 Teams based on Athlete Count</h1>',
        unsafe_allow_html=True
    )
    #st.title("Top 10 Teams based on Athlete Count")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=df["Team"].value_counts().head(10).index,
        y=df["Team"].value_counts().head(10).values,
        palette = "husl"
    )
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and horizontal alignment as needed
    plt.xlabel("Team")
    plt.ylabel("Athlete Count")
    plt.title("Top 10 Teams based on Athlete Count")
    st.pyplot(plt)

    st.title(selected_country+" Medal Tallys Over The Years")

    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt=helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 Athletes Of "+selected_country)
    top10_df=helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athlete-Wise Analysis':

    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()
    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)

    st.markdown(
        '<h1 style="color: #800080;">Distribution of Age</h1>',
        unsafe_allow_html=True)
    #st.title("Distribution of Age")
    st.plotly_chart(fig)
    st.markdown(
        '<h1 style="color: #800080;">Overall Age Distribution of Participants</h1>',
        unsafe_allow_html=True)
    #st.title("Overall Age Distribution of Participants")
    plt.figure(figsize=(12, 6))
    plt.title('Overall Age distribution of the participants')
    plt.xlabel('Age')
    plt.ylabel('Number of participants')
    plt.hist(df.Age, bins=np.arange(10, 80, 2), color='orange', edgecolor='white')
    st.pyplot(plt)
    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.markdown(
        '<h1 style="color: #800080;">Distribution of Age wrt Sports(Gold Medalist)</h1>',
        unsafe_allow_html=True)
    #st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.markdown(
        '<h1 style="color: #800080;">Height Vs Weight</h1>',
        unsafe_allow_html=True)
    #st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df=helper.weight_v_height(df,selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'],y=temp_df['Height'],hue=temp_df['Medal'],style=temp_df["Sex"],s=60)
    plt.xlabel("Weight")
    plt.ylabel("Height")
    st.pyplot(fig)

    st.markdown(
        '<h1 style="color: #800080;">Men Vs Women Participation Over the Years</h1>',
        unsafe_allow_html=True)
    #st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)
    womenOlympics = df[(df.Sex == 'F') & (df.Season == 'Summer')]
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10))
    sns.countplot(x='Year', data=womenOlympics, palette="Spectral")
    st.markdown(
        '<h1 style="color: #800080;">Women Participation in Olympics Over the Years</h1>',
        unsafe_allow_html=True)
    #plt.title('Women Participation in Summer Olympics Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    st.pyplot(plt)
    menOlympics = df[(df.Sex == 'M') & (df.Season == 'Summer')]
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10))
    sns.countplot(x='Year', data=menOlympics, palette="Spectral")
    st.markdown(
        '<h1 style="color: #800080;">Men Participation in Olympics Over the Years</h1>',
        unsafe_allow_html=True)
    #plt.title('Men Participation in Summer Olympics Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    st.pyplot(plt)
    st.markdown(
        '<h1 style="color: #800080;">Gender Distribution of Athletes</h1>',
        unsafe_allow_html=True)
    #st.title("Gender Distribution of Athletes")
    gender_counts = df['Sex'].value_counts()
    fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
    st.plotly_chart(fig)

    st.markdown('<h1 style="color: #800080;">Number Of Athletes participated in Particular Sport in Particular Year</h1>', unsafe_allow_html=True)
    # Define the time intervals
    intervals = [(1996, 2016), (1976, 1996), (1956, 1976), (1936, 1956), (1916, 1936), (1896, 1916)]
    # Define a list of colors for each subplot
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Flatten the 2x3 grid to 1D
    axes = axes.flatten()
    for i, (start_year, end_year) in enumerate(intervals, start=1):
        ax = axes[i - 1]
        counts = pd.value_counts(df[(df['Year'] < end_year) & (df['Year'] > start_year)].Sport)[:10]
        counts.plot(kind='bar', color=colors[i - 1], ax=ax)
        ax.set_title(f'{start_year} - {end_year}')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot in Streamlit
    st.pyplot(fig)


if user_menu=="Medal-Wise Analysis":
    st.title("Overall Top 10 Medal Winning Regions")
    df["medals_count"] = df.Medal.map(lambda x: 1 if type(x) == str else 0)
    medals_count_by_region = df['medals_count'].groupby(df['region']).sum().sort_values(ascending=False).head(10)
    colors = sns.color_palette("Set2", n_colors=len(medals_count_by_region))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=medals_count_by_region.index, y=medals_count_by_region.values, palette=colors)
    plt.xlabel('Region')
    plt.ylabel('Number of Won Medals')
    plt.title('Top 10 Medals Winning Regions')
    st.pyplot(plt)

    st.title("Top 10 Gold Medal Winning Regions")
    gold_medals = df[df["Medal"] == "Gold"]["region"].value_counts().sort_values(ascending=False).head(10)
    colors = sns.color_palette("coolwarm", n_colors=len(gold_medals))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=gold_medals.index, y=gold_medals.values, palette = colors)
    plt.xlabel('Region')
    plt.ylabel('Gold Medals')
    plt.title('Top 10 Gold Medal Winning Regions')
    st.pyplot(plt)

    st.title("Top 10 Silver Medal Winning Regions")
    silver_medals = df[df["Medal"] == "Silver"]["region"].value_counts().sort_values(ascending=False).head(10)
    colors = sns.color_palette("magma", n_colors=len(silver_medals))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=silver_medals.index, y=silver_medals.values, palette=colors)
    plt.xlabel('Region')
    plt.ylabel('Silver Medals')
    plt.title('Top 10 Silver Medal Winning Regions')
    st.pyplot(plt)

    st.title("Top 10 Bronze Medal Winning Regions")
    bronze_medals = df[df["Medal"] == "Bronze"]["region"].value_counts().sort_values(ascending=False).head(10)
    colors = sns.color_palette("viridis", n_colors=len(bronze_medals))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=bronze_medals.index, y=bronze_medals.values, palette=colors)
    plt.xlabel('Region')
    plt.ylabel('Bronze Medals')
    plt.title('Top 10 Bronze Medal Winning Regions')
    st.pyplot(plt)

    st.title("Medals for Top 20 countries")

    total_medals_by_country = df[df['Season'] == 'Summer'].groupby(['region', 'Medal'])['Sex'].count().reset_index()
    total_medals_by_country = total_medals_by_country.pivot_table(index='region', columns='Medal', values='Sex',
                                                                  fill_value=0).sort_values(by='Gold',
                                                                                            ascending=False).head(20)
    # Plotting
    total_medals_by_country.plot(kind='bar')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel("Athlete count")
    plt.title('Medals by Country - Summer Olympics')
    # Display the plot using Streamlit
    st.pyplot(plt)

    st.title("Which Athele won the most medal ?")
    df["test"] = df.Medal.map(lambda x: 1 if type(x) == str else 0)
    medals_count_by_athlete = df['test'].groupby(df['Name']).sum().sort_values(ascending=False).head(10)
    colors = sns.color_palette("muted", n_colors=len(medals_count_by_athlete))
    plt.figure(figsize=(8, 8))
    plt.pie(medals_count_by_athlete, labels=medals_count_by_athlete.index, autopct='%1.1f%%', colors=colors)
    plt.title('Top 10 Athletes and Medals Investigation')
    st.pyplot(plt)

    df["test"] = df.Medal.map(lambda x: 1 if type(x) == str else 0)
    medals_count_by_athlete = df['test'].groupby(df['Name']).sum().sort_values(ascending=False).head(10)
    colors = sns.color_palette("muted", n_colors=len(medals_count_by_athlete))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=medals_count_by_athlete.values, y=medals_count_by_athlete.index, palette=colors)
    plt.ylabel('Player')
    plt.xlabel('Number of Medals')
    plt.title('Top 10 Athletes and Medals Investigation')
    st.pyplot(plt)

if user_menu == "About":

    lt = ["Identified the most successful countries and athletes in terms of medal count.",
          "Performed deeper analyses on specific countries or athletes to explore their performance over time.",
          "Determined if certain countries or athletes have dominant sports where they excel.",
          "Plotted trends over time, such as the number of participants, gender ratio, or medal distributions.",
          "Visualized the success of countries or athletes based on the number of medals won.",
          ]
    st.header("120 years of Olympic history: athletes and results")

    st.markdown("For this Web Application , We have used a historical dataset on the modern Olympic Games,"
                "including all the Games from Athens 1896 to Rio 2016. We got this data set from Kaggle whose link is given below:")
    st.write(
        "Dataset : [Link](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)")
    st.header("About Olympics Data Analysis:")
    st.write("We have,")
    for item in lt:
        st.markdown("* " + item)
    st.header("About Olympics Medal Predictor:")
    selected_col = ["Sex", "Region", "Sport", "Height", "Weight", "Age"]
    st.write("We made a Olympic Medal predictor which can predict the possibility of an athlete",
             "winning an Olympics Medal on the basis of his/her: " + str(selected_col))
    # for item in selected_col:
    # st.markdown(selected_col)
    st.write(
        "In this Medal Predictor, We have used two Maching Learning algorithms and a Artificial Neural Network (Deep Learning) namely:")
    model = ["Random Forest Classifier", "Logistic Regression", "Neutral Network"]
    for item in model:
        st.markdown("* " + item)
modelrfc = pickle.load(open("C:/Users/MA/Downloads/modelrfc.pkl","rb"))
modellr = pickle.load(open("C:/Users/MA/Downloads/modellr.pkl","rb"))
transformer = pickle.load(open("C:/Users/MA/Downloads/transformer.pkl","rb"))
#modelln=pickle.load(open("C:/Users/MA/Downloads/modelnn.pkl","rb"))

if user_menu == "Medal Predictor":
    st.title("Olympics Medal Predictor")
    selected_col = ["Sex", "region", "Sport", "Height", "Weight", "Age"]
    sport = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton',
             'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing',
             'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism',
             'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey',
             'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon',
             'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby',
             'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping',
             'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis',
             'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo',
             'Weightlifting', 'Wrestling']
    country = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina',
               'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
               'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina',
               'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
               'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
               'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',
               'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
               'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland',
               'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala',
               'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India',
               'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast',
               'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
               'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
               'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
               'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
               'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
               'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea',
               'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo',
               'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino',
               'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia',
               'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan',
               'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania',
               'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA',
               'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
               'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']
    with st.form("my_form"):
        Sex = st.selectbox("Select Sex", ["M", "F"])
        Age = st.slider("Select Age", 10, 97)
        Height = st.slider("Select Height(In centimeters)", 127, 226)
        Weight = st.slider("Select Weight(In kilograms)", 25, 214)
        region = st.selectbox("Select Country", country)
        Sport = st.selectbox("Select Sport", sport)
        input_model = st.selectbox("Select Prediction Model",
                                   ["Random Forest Classifier", "Logistic Regression", "Neutral Network"])

        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs = [Sex, region, Sport, Height, Weight, Age]
            inputs = pd.DataFrame([inputs], columns=selected_col)
            inputs = transformer.transform(inputs)
            if input_model == "Random Forest Classifier":
                model = modelrfc
            if input_model == "Logistic Regression":
                model = modellr
            if input_model == "Neutral Network":
                model = modelrfc
            prediction = model.predict(inputs)
            with st.spinner('Predicting output...'):
                time.sleep(1)
                if prediction[0] == 0:
                    ans = "Low"
                    st.warning("Medal winning probability is {}".format(ans), icon="⚠️")
                else:
                    ans = "High"
                    st.success("Medal winning probability is {}".format(ans), icon="✅")

if user_menu=="Match Prediction":
    st.title("Olympics Medal Predictor for Next Olympics")
    sport = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton',
             'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing',
             'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism',
             'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey',
             'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon',
             'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby',
             'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping',
             'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis',
             'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo',
             'Weightlifting', 'Wrestling']
    country = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina',
               'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
               'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina',
               'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
               'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
               'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',
               'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
               'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland',
               'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala',
               'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India',
               'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast',
               'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
               'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
               'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
               'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
               'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
               'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea',
               'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo',
               'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino',
               'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia',
               'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan',
               'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania',
               'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA',
               'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
               'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']

    # Collect user input
    Sex = st.selectbox("Select Sex", ["M", "F"])
    Age = st.slider("Select Age", 10, 97)
    Height = st.slider("Select Height(In centimeters)", 127, 226)
    Weight = st.slider("Select Weight(In kilograms)", 25, 214)
    region = st.selectbox("Select Country", country)
    Sport = st.selectbox("Select Sport", sport)
    selected_col = ["Sex", "region", "Sport", "Height", "Weight", "Age"]

    # Prepare input data for prediction
    inputs = pd.DataFrame([[Sex, region, Sport, Height, Weight, Age]], columns=selected_col)
    inputs = transformer.transform(inputs)

    # Make predictions
    pred_rfc = modelrfc.predict(inputs)
    pred_lr = modellr.predict(inputs)

    # Concatenate predictions from both models
    medal_probabilities = np.concatenate([pred_rfc, pred_lr, pred_rfc])

    # Display predicted medal counts or probabilities
    st.write(f"Predicted Medal Probabilities for {region} in Next Olympics:")
    st.write("Gold:", medal_probabilities[0])
    st.write("Silver:", medal_probabilities[1])
    st.write("Bronze:", medal_probabilities[2])



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

if user_menu == 'Cluster Analysis':
    st.sidebar.title('Cluster Analysis')
    st.sidebar.write('Perform clustering to identify patterns in the data.')

    features_for_clustering = st.multiselect('Select features for clustering:', df.select_dtypes(include=['float64', 'int64']).columns)

    if features_for_clustering:
        st.write(f'Clustering based on features: {", ".join(features_for_clustering)}')

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        df[features_for_clustering] = imputer.fit_transform(df[features_for_clustering])

        # Perform clustering
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features_for_clustering])
        k = st.slider('Select the number of clusters (K) for K-Means:', 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)

        # Determine valid n_components for PCA
        n_components = min(df_scaled.shape[0], len(features_for_clustering))

        # Visualize the clusters using PCA (dimensionality reduction)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df_scaled)

        # Create new PCA column(s) based on the number of components
        for i in range(n_components):
            df[f'PCA{i + 1}'] = pca_result[:, i]

        # Scatter plot of clusters
        fig = px.scatter(df, x=f'PCA1', y=f'PCA2', color='Cluster', title='Cluster Analysis', labels={'Cluster': 'Cluster'})
        st.plotly_chart(fig)

        st.write('Cluster Centers:')
        st.write(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features_for_clustering))

    else:
        st.warning('Please select at least one numerical feature for clustering.')
