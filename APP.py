from sklearn.utils import resample

import streamlit as st

st.markdown(
    """
    <style>
    /* Style pour personnaliser l'apparence de la barre latérale */
    .css-1d391kg { /* Cette classe représente la barre latérale */
        background-color: #ADD8E6;  /* Change ici la couleur d'arrière-plan de la barre latérale */
        border: 1px solid #2E8B57;  /* Optionnel : ajouter une bordure */
    }

    /* Changer la couleur du texte dans la barre latérale */
    .css-1d391kg .css-3o5yew {
        color: #ADD8E6;  /* Change ici la couleur du texte des éléments dans le sommaire */
    }

    /* Changer la couleur des titres dans la barre latérale (comme "Sommaire") */
    .css-1hynsf2 { 
        color: #ADD8E6;  /* Change ici la couleur du texte des titres */
    }
    </style>
    """,
    unsafe_allow_html=True

)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import plotly.express as px


from scipy.stats import skew, kurtosis



# Importation des modèles nécessaires
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Importation des métriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import streamlit as st
import plotly.graph_objects as go





import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LogisticRegression


from scipy.stats import pearsonr, chi2_contingency, ttest_ind, f_oneway


from scipy.stats import pearsonr, chi2_contingency, f_oneway



# Charger les données de la base

df = pd.read_csv("heart_diseases.csv", sep =",")

df_Modelisation = pd.read_csv("/Users/oumarkante/Desktop/SPARK/df_Modelisation")







descriptions = {
    "State": "Code FIPS de l'État",
    "Sex": "Sexe du répondant",
    "GeneralHealth": "Diriez-vous qu'en général, votre santé est",
    "PhysicalHealthDays": "En pensant à votre santé physique, qui comprend les maladies et les blessures physiques, combien de jours au cours des 30 derniers jours votre santé physique n'a-t-elle pas été bonne ?",
    "MentalHealthDays": "En ce qui concerne votre santé mentale, qui comprend le stress, la dépression et les problèmes émotionnels, combien de jours au cours des 30 derniers jours votre santé mentale n'a-t-elle pas été bonne ?",
    "LastCheckupTime": "Depuis combien de temps environ n'avez-vous pas consulté un médecin pour un contrôle de routine ?",
    "PhysicalActivities": "Au cours du dernier mois, en dehors de votre travail habituel, avez-vous participé à des activités physiques ou à des exercices tels que la course à pied, la gymnastique suédoise, le golf, le jardinage ou la marche pour faire de l'exercice ?",
    "SleepHours": "En moyenne, combien d'heures de sommeil avez-vous par période de 24 heures ?",
    "RemovedTeeth": "Sans compter les dents perdues pour cause de blessure ou d'orthodontie, combien de vos dents permanentes ont été enlevées à cause d'une carie dentaire ou d'une maladie des gencives ?",
    "HadHeartAttack": "(On vous a déjà dit) que vous aviez eu une crise cardiaque, également appelée infarctus du myocarde ?",
    "HadAngina": "(On vous a déjà dit) que vous souffriez d'angine de poitrine ou d'une maladie coronarienne ?",
    "HadStroke": "(On vous l'a dit) (vous avez eu) un accident vasculaire cérébral.",
    "HadAsthma": "(On vous a déjà dit) (que vous aviez) de l'asthme ?",
    "HadSkinCancer": "(On vous a dit) que vous aviez un cancer de la peau qui n'est pas un mélanome ?",
    "HadCOPD": "(On vous a déjà dit) que vous souffriez d'une maladie pulmonaire obstructive chronique, d'emphysème ou de bronchite chronique ?",
    "HadDepressiveDisorder": "(On vous a déjà dit) (que vous aviez) un trouble dépressif (y compris la dépression, la dépression majeure, la dysthymie ou la dépression mineure) ?",
    "HadKidneyDisease": "A l'exception des calculs rénaux, des infections de la vessie ou de l'incontinence, vous a-t-on déjà dit que vous souffriez d'une maladie rénale ?",
    "HadArthritis": "(On vous a déjà dit) (que vous aviez) une forme d'arthrite, de polyarthrite rhumatoïde, de goutte, de lupus ou de fibromyalgie ?  (Les diagnostics d'arthrite comprennent : rhumatisme, polymyalgie rhumatismale ; arthrose (pas ostéporose) ; tendinite, bursite, oignon, tennis elbow ; syndrome du canal carpien, syndrome du canal tarsien ; infection articulaire, etc.)",
    "MentalHealthDays": "....",
    "HadDiabetes": "(On vous a déjà dit) (que vous aviez) du diabète ?",
    "DeafOrHardOfHearing": "Êtes-vous sourd ou avez-vous de sérieuses difficultés à entendre ?",
    "BlindOrVisionDifficulty": "Êtes-vous aveugle ou avez-vous de sérieuses difficultés à voir, même en portant des lunettes ?",
    "DifficultyConcentrating": "En raison d'un état physique, mental ou émotionnel, avez-vous de sérieuses difficultés à vous concentrer, à vous souvenir ou à prendre des décisions ?",
    "DifficultyWalking": "Avez-vous de sérieuses difficultés à marcher ou à monter les escaliers ?",
    "DifficultyDressingBathing": "Avez-vous des difficultés à vous habiller ou à vous laver ?",
    "DifficultyErrands": "En raison d'un état physique, mental ou émotionnel, avez-vous des difficultés à faire des courses seul, comme aller chez le médecin ou faire des achats ?",
    "SmokerStatus": "Statut de fumeur à quatre niveaux :  Fumeur quotidien, Fumeur occasionnel, Ancien fumeur, Non-fumeur",
    "ECigaretteUsage": "Diriez-vous que vous n'avez jamais utilisé d'e-cigarettes ou d'autres produits de vapotage électronique au cours de votre vie, que vous les utilisez tous les jours, que vous les utilisez de temps en temps ou que vous les avez utilisés dans le passé mais que vous ne les utilisez plus du tout à l'heure actuelle ?",
    "ChestScan": "Avez-vous déjà subi un scanner ou une tomodensitométrie de votre région thoracique ?",
    "RaceEthnicityCategory": "Catégorie de race/ethnie à cinq niveaux",
    "AgeCategory": "Catégorie d'âge à quatorze niveaux",
    "HeightInMeters": "Taille déclarée en mètres",
    "WeightInKilograms": "Poids déclaré en kilogrammes",
    "BMI": "Indice de masse corporelle (IMC)",
    "AlcoholDrinkers": "Adultes ayant déclaré avoir bu au moins un verre d'alcool au cours des 30 derniers jours.",
    "HIVTesting": "Adultes ayant déjà subi un test de dépistage du VIH",
    "FluVaxLast12": "Au cours des 12 derniers mois, avez-vous reçu un vaccin antigrippal par pulvérisation dans le nez ou par injection dans le bras ?",
    "PneumoVaxEver": "Avez-vous déjà reçu un vaccin contre la pneumonie, également connu sous le nom de vaccin antipneumococcique ?",
    "TetanusLast10Tdap": "Avez-vous reçu un vaccin contre le tétanos au cours des 10 dernières années ? S'agissait-il du Tdap, le vaccin contre le tétanos qui contient également le vaccin contre la coqueluche ?",
    "HighRiskLastYear": "Au cours de l'année écoulée, vous vous êtes injecté des drogues autres que celles qui vous ont été prescrites. Vous avez été traité(e) pour une maladie sexuellement transmissible (MST) au cours de l'année écoulée. Vous avez donné ou reçu de l'argent ou de la drogue en échange de relations sexuelles au cours de l'année écoulée.",
    "CovidPos": "Un médecin, une infirmière ou un autre professionnel de la santé vous a-t-il déjà dit que vous aviez été testé(e) positif(ve) au COVID 19 ?"
}





# Créer des pages
st.sidebar.title("Sommaire")
pages = ["Contexte du projet","Exploration des données","Description des variables",  "Analyse de données", "Modélisation et Prédiction"]
page = st.sidebar.radio("Aller vers la page :", pages)









# Page 1: Contexte du projet
if page == pages[0]:
    # Bannière d'introduction avec un texte accrocheur
    st.markdown("""
        <div style="background-color:#2E8B57;padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">📈 Projet: Analyse Big Data et Data Mining pour la Prediction des Maladies Cardiaques</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write(" ")

    # Présentation avec un résumé stylisé
    st.markdown("""
    <div style="border-left: 6px solid #2E8B57; padding-left: 15px; margin: 20px 0; font-size: 16px;">
        <p style="font-family: Arial, sans-serif; font-size: 18px;">
            Ce projet vise à exploiter les <b>données massives</b> pour prédire la probabilité qu'une personne souffre d'une <b>maladie cardiaque</b>. 
        L'objectif est de développer des <i>modèles de machine learning</i> permettant d'identifier les facteurs de risque critiques et d'améliorer les diagnostics pr\u00e9dictifs.
    </p>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write(" ")

    # Objectifs du Projet avec décorations
    st.subheader("🎯 Objectifs du Projet")

    # Créer une liste des objectifs avec des descriptions détaillées
    objectives = {
        "📝 Description des Variables": "Analyser les différentes caractéristiques personnelles et historiques médicales des patients, comme l'âge, l'IMC, les habitudes de tabagisme, et les niveaux d'activité physique.",
        "🔍 Exploration des Données": "Examiner les données pour détecter des patterns, des corrélations, ainsi que pour identifier les anomalies ou les valeurs manquantes.",
        "📊 Analyse des Données": "Comprendre les facteurs de risque et identifier les indicateurs clés associés aux maladies cardiaques.",
        "🤖 Modélisation": "Construire et tester des modèles prédictifs pour estimer la probabilité qu'une personne souffre d'une maladie cardiaque, en utilisant des algorithmes comme la régression logistique, Random Forest et Gradient Boosting.."
    }

    for objective, description in objectives.items():
        st.markdown(f"""
        <div style="border: 1px solid #2E8B57; padding: 15px; border-radius: 8px; margin: 10px 0; background-color: #F9F9F9;">
            <h4 style="color: #2E8B57; margin-bottom: 5px;">{objective}</h4>
            <p style="font-family: Arial, sans-serif; color: #555;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    # Image avec légende et séparation visuelle
    st.image("photo.jpg", caption="Illustration d'une campagne de marketing direct", use_column_width=True)
    st.markdown("<hr style='border:1px solid #2E8B57'>", unsafe_allow_html=True)




















if 'df' not in st.session_state:
    st.session_state.df = df








elif page == pages[1]:
    st.write("### Exploration des données")

    # Utiliser st.session_state.df
    df = st.session_state.df

    # Aperçu des données
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # Dimensions du dataframe
    if st.checkbox("Afficher les dimensions du dataframe 📏"):
        st.write(f"**Dimensions :** Nous avons un total de {df.shape[0]} répondants et {df.shape[1]} variables à analyser.")

    # Valeurs manquantes
    if st.checkbox("Afficher les valeurs manquantes ❓"):
        missing_values = df.isna().sum()
        st.write("### Valeurs manquantes :")
        st.dataframe(missing_values[missing_values > 0])  # Afficher uniquement les colonnes avec des valeurs manquantes
        if missing_values.sum() == 0:
            st.write("Nous avons **zéro (0)** valeur manquante dans le dataset. 🎉")
        else:
            st.write(f"Nous avons un total de **{missing_values.sum()}** valeurs manquantes dans le dataset.")

    # Gestion des doublons
    if st.checkbox("Afficher les doublons 🔍"):
        num_duplicates = df.duplicated().sum()
        st.write(f"### Nombre de doublons : {num_duplicates}")
        if num_duplicates == 0:
            st.write("Nous avons **zéro (0)** doublon dans le dataset. 🎉")
        else:
            st.write(f"Nous avons un total de **{num_duplicates}** doublons dans le dataset.")
            
            # Ajouter un bouton pour supprimer les doublons
            if st.button("Supprimer les doublons"):
                st.session_state.df = st.session_state.df.drop_duplicates()  # Mise à jour dans st.session_state
                st.success("Doublons supprimés avec succès !")
                st.write("Données après suppression des doublons :")
                st.dataframe(st.session_state.df)

    # Détection des valeurs aberrantes
    if st.checkbox("Vérifier les valeurs aberrantes 🧹"):
        # Détection des colonnes numériques
        numeric_columns = df.select_dtypes(include=['number']).columns
        outlier_info = {}

        # Identifier les valeurs aberrantes
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            num_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

            if num_outliers > 0:
                outlier_info[col] = num_outliers

        # Affichage des résultats
        if not outlier_info:
            st.write("Aucune valeur aberrante détectée dans les colonnes numériques. 🎉")
        else:
            st.write("### Résumé des valeurs aberrantes détectées :")
            for col, count in outlier_info.items():
                st.write(f"- **{col}** : {count} valeur(s) aberrante(s) détectée(s).")

            # Bouton pour supprimer les valeurs aberrantes
            if st.button("Supprimer les valeurs aberrantes"):
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                st.session_state.df = df  # Mise à jour dans st.session_state
                st.success("Valeurs aberrantes supprimées avec succès !")
                st.write("Données après suppression des valeurs aberrantes :")
                st.dataframe(st.session_state.df)















elif page == pages[2]:
    st.write("### 🧐 Description des variables")

    # Utiliser st.session_state.df
    df = st.session_state.df

     
    
    # Sélection de la variable
    selected_variable = st.selectbox("Choisissez une variable pour voir sa signification :", list(descriptions.keys()))
    st.write(f"**{selected_variable}** : {descriptions[selected_variable]}")

    # Séparateur
    st.markdown("<hr>", unsafe_allow_html=True)

    # Vérification du type de variable (numérique ou catégorielle)
    if pd.api.types.is_numeric_dtype(df[selected_variable]):
        # Variables numériques
        x = df[selected_variable]

        # Histogramme avec courbe de densité normale
        x_density = np.linspace(min(x), max(x), 1000)
        hist_chart = go.Figure()
        hist_chart.add_trace(go.Histogram(x=x, histnorm='probability density', nbinsx=20, name='Histogramme', marker_color='#1f77b4'))
        hist_chart.add_trace(go.Scatter(
            x=x_density,
            y=np.exp(-0.5 * ((x_density - np.mean(x)) / np.std(x))**2) / (np.std(x) * np.sqrt(2 * np.pi)),
            mode='lines', name='Courbe Normale', line=dict(color='#ff7f0e', width=2)
        ))

        # Boxplot
        boxplot_chart = go.Figure()
        boxplot_chart.add_trace(go.Box(y=x, boxmean="sd", name=f"Boxplot de {selected_variable}", marker_color='#2ca02c'))
        boxplot_chart.update_layout(title=f"Boxplot de {selected_variable}", title_x=0.5)



        # Affichage des graphiques
        st.plotly_chart(hist_chart, use_container_width=True)
        st.plotly_chart(boxplot_chart, use_container_width=True)


        # Calcul de la skewness et kurtosis
        skewness = skew(x)
        kurt = kurtosis(x)

        # Calcul des statistiques descriptives
        stats = x.describe()

        # Ajout de la skewness et kurtosis aux statistiques
        stats['skewness'] = skewness
        stats['kurtosis'] = kurt

        # Convertir les statistiques en DataFrame pour appliquer des styles
        stats_df = stats.to_frame().T  # Convertir en DataFrame (ajouter une ligne)

        # Statistiques descriptives avec mise en forme
        st.markdown("### 📊 Statistiques descriptives")
        st.dataframe(stats_df.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='salmon'))

        # Séparateur
        st.markdown("<hr>", unsafe_allow_html=True)

        # Analyse concise
        analyse_auto = f"""
        La variable **{selected_variable}** a une moyenne de {stats['mean']:.2f}, indiquant la valeur centrale des données. 
        L'écart-type de {stats['std']:.2f} montre la dispersion des données. 
        La médiane de {stats['50%']:.2f} est proche de la moyenne, ce qui suggère une distribution relativement symétrique.
        La skewness de {skewness:.2f} indique un léger décalage de la distribution, et la kurtosis de {kurt:.2f} montre si la distribution est plus ou moins concentrée que la normale.
        """
        if st.checkbox("Analyse automatique", key="analyse_auto"):
            st.write(f"**Analyse pour {selected_variable} :**\n{analyse_auto}")

        # Séparateur
        st.markdown("<hr>", unsafe_allow_html=True)

        # Interprétation concise
        interpretation_auto = f"""
        En résumé, **{selected_variable}** présente une distribution relativement symétrique avec un léger décalage vers les valeurs élevées (skewness {skewness:.2f}). 
        L'écart-type indique une dispersion modérée des données autour de la moyenne. 
        Les valeurs aberrantes peuvent être identifiées sur le boxplot.
        """
        if st.checkbox("Interprétation automatique", key="interpretation_auto"):
            st.write(f"**Interprétation pour {selected_variable} :**\n{interpretation_auto}")

        
    else:
        # Variables catégorielles
        counts = df[selected_variable].value_counts()
        percentages = df[selected_variable].value_counts(normalize=True) * 100

        # Graphique circulaire
        pie_chart = px.pie(
            df,
            names=selected_variable,
            title=f"Répartition de {selected_variable}",
            hole=0.3,  # Pour un style en anneau
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(pie_chart, use_container_width=True)

        # Séparateur
        st.markdown("<hr>", unsafe_allow_html=True)

        # Analyse concise des catégories
        analyse_auto_cat = f"La variable **{selected_variable}** présente les catégories suivantes :\n"
        for category, count in counts.items():
            analyse_auto_cat += f"- **{category}** : {count} occurrences ({percentages[category]:.1f}%)\n"
        
        if st.checkbox("Analyse automatique", key="analyse_auto_cat"):
            st.write(f"**Analyse pour {selected_variable} :**\n{analyse_auto_cat}")

        # Séparateur
        st.markdown("<hr>", unsafe_allow_html=True)

        # Interprétation concise pour les variables catégorielles
        interpretation_auto_cat = f"""
        **{selected_variable}** est déséquilibrée, avec une catégorie dominante (**{counts.idxmax()}**) représentant {percentages.max():.1f}% des données. 
        Ce déséquilibre pourrait influencer certaines analyses, et il peut être nécessaire d'appliquer des techniques de rééchantillonnage pour traiter ce biais.
        """
        if st.checkbox("Interprétation automatique", key="interpretation_auto_cat"):
            st.write(f"**Interprétation pour {selected_variable} :**\n{interpretation_auto_cat}")






















# Page 4: Analyse Bivariée entre les Variables et HadHeartAttack (avec HadHeartAttack qualitative)
elif page == pages[3]:
    st.write("### Analyse de données")

    # Utiliser st.session_state.df
    df = st.session_state.df

     


    # Analyse Bivariée avec la variable cible 'HadHeartAttack'
    st.subheader("Analyse Bivariée avec la Variable Cible")
    st.write("Choisissez une variable à analyser par rapport à la variable cible `HadHeartAttack` :")

    # Liste déroulante pour sélectionner la variable à comparer avec 'HadHeartAttack'
    variable_bivarie = st.selectbox("Choisissez une variable :", [col for col in df.columns if col != 'HadHeartAttack' and col != 'id'])

    # Vérification du type de la variable sélectionnée
    if variable_bivarie in df.select_dtypes(include=['float64', 'int64']).columns:
        # Analyse pour les variables numériques
        st.write(f"### Répartition de `{variable_bivarie}` selon la variable cible `HadHeartAttack`")
        st.write(df.groupby('HadHeartAttack')[variable_bivarie].describe())  # Affichage des stats descriptives par groupe

        # Visualisation : Boîte à moustaches pour comparer les distributions
        fig = px.box(df, x='HadHeartAttack', y=variable_bivarie, title=f"Distribution de `{variable_bivarie}` par rapport à `HadHeartAttack`")
        st.plotly_chart(fig)

        # Test t de Student ou ANOVA
        unique_categories = df["HadHeartAttack"].nunique()  # Nombre de catégories dans 'HadHeartAttack'

        if unique_categories == 2:
            # Si `HadHeartAttack` a 2 catégories : Test t de Student
            st.write(f"### Test t de Student pour `{variable_bivarie}` en fonction de `HadHeartAttack`")

            # Séparation des données en 2 groupes
            group1 = df[df["HadHeartAttack"] == df["HadHeartAttack"].unique()[0]][variable_bivarie].dropna()
            group2 = df[df["HadHeartAttack"] == df["HadHeartAttack"].unique()[1]][variable_bivarie].dropna()

            # Test t de Student
            t_stat, p_value = ttest_ind(group1, group2)
            st.write(f"**Statistique t** : {t_stat:.4f}")
            st.write(f"**Valeur p** : {p_value:.4f}")

            if p_value < 0.05:
                st.write(f"Il existe une différence significative entre les groupes pour `{variable_bivarie}` et `HadHeartAttack`.")
            else:
                st.write(f"Aucune différence significative entre les groupes pour `{variable_bivarie}` et `HadHeartAttack`.")

        else:
            # Si `y` a plus de 2 catégories : Test ANOVA
            st.write(f"### Test ANOVA pour `{variable_bivarie}` en fonction de `HadHeartAttack`")

            # Séparation des données en groupes en fonction des catégories de `y`
            groups = [df[df["HadHeartAttack"] == category][variable_bivarie].dropna() for category in df["HadHeartAttack"].unique()]

            # Test ANOVA
            f_stat, p_value = f_oneway(*groups)
            st.write(f"**Statistique F** : {f_stat:.4f}")
            st.write(f"**Valeur p** : {p_value:.4f}")

            if p_value < 0.05:
                st.write(f"Il existe une différence significative entre les groupes pour `{variable_bivarie}` et `HadHeartAttack`.")
            else:
                st.write(f"Aucune différence significative entre les groupes pour `{variable_bivarie}` et `HadHeartAttack`.")

    elif variable_bivarie in df.select_dtypes(include=['object']).columns:
        # Analyse pour les variables catégorielles
        st.write(f"### Distribution de `{variable_bivarie}` par rapport à `HadHeartAttack`")
        contingency_table = pd.crosstab(df[variable_bivarie], df['HadHeartAttack'])
        st.dataframe(contingency_table)

        # Visualisation : Histogramme empilé
        fig = go.Figure(data=[
            go.Bar(name='Yes', x=contingency_table.index, y=contingency_table['Yes']),
            go.Bar(name='No', x=contingency_table.index, y=contingency_table['No'])
        ])
        fig.update_layout(barmode='stack', title=f"Distribution de `{variable_bivarie}` par rapport à `HadHeartAttack`")
        st.plotly_chart(fig)

        # Test du Chi-Carré
        st.write(f"### Test du Chi-Carré pour `{variable_bivarie}` et `HadHeartAttack`")
        
        # Création de la table de contingence
        contingency_table = pd.crosstab(df[variable_bivarie], df['HadHeartAttack'])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        st.write(f"**Statistique du Chi-Carré** : {chi2_stat:.4f}")
        st.write(f"**Valeur p** : {p_value:.4f}")

        if p_value < 0.05:
            st.write(f"Il existe une relation significative entre `{variable_bivarie}` et `HadHeartAttack`.")
        else:
            st.write(f"Aucune relation significative entre `{variable_bivarie}` et `HadHeartAttack`.")





# Analyse de Corrélation
    st.subheader("Analyse de Corrélation")
    st.write("Exploration des relations entre variables numériques.")
    
    if st.checkbox("Afficher la matrice de corrélation 📊"):
        correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
        st.dataframe(correlation_matrix)

        # Visualisation du Heatmap des Corrélations
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation'),
            text=correlation_matrix.values,
            hoverinfo='text'
        ))
        fig.update_layout(title="Heatmap de Corrélation")
        st.plotly_chart(fig)



     

    # Analyse Avancée - Nuage de Points Interactif
    st.write("### Analyse Avancée : Nuage de Points ")

    # Choix des variables numériques pour les axes
    variable_x = st.selectbox("Choisissez une variable pour l'axe X :", df.select_dtypes(include=['float64', 'int64']).columns)
    variable_y = st.selectbox("Choisissez une variable pour l'axe Y :", df.select_dtypes(include=['float64', 'int64']).columns)

    # Option pour ajouter une variable catégorielle pour la couleur
    variable_hue = st.selectbox("Choisissez une variable catégorielle pour la couleur (optionnel) :", 
                            [None] + list(df.select_dtypes(include=['object']).columns))

    # Création du nuage de points interactif avec Plotly
    fig = px.scatter(
        df, 
        x=variable_x, 
        y=variable_y, 
        color=variable_hue, 
        title=f"Nuage de Points  : {variable_x} vs {variable_y}",
        labels={variable_x: variable_x, variable_y: variable_y},
        template="plotly_white"
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)



    

















# elif page == pages[4]:
#     st.write("### Modélisation: regression logistique  ")

#     # Chargement des données
#     df_Modelisation = df_Modelisation
#     target = 'HadHeartAttack'  # Variable cible
#     X = df_Modelisation.drop(columns=[target])
#     y = df_Modelisation[target]

#     # Conversion de la variable cible en binaire (1 pour 'Yes', 0 pour 'No')
#     y = y.map({'Yes': 1, 'No': 0})

#     # Sélection des variables
#     quantitative_vars = X.select_dtypes(include=['number']).columns
#     qualitative_vars = X.select_dtypes(exclude=['number']).columns

#     selected_quant_vars = st.multiselect("Sélectionner les variables quantitatives", quantitative_vars, default=quantitative_vars)
#     selected_qual_vars = st.multiselect("Sélectionner les variables qualitatives", qualitative_vars, default=qualitative_vars)

#     X_selected = X[selected_quant_vars + selected_qual_vars]

#     # Traitement des variables
#     quant_pipeline = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler())
#     ])
#     qual_pipeline = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('quant', quant_pipeline, selected_quant_vars),
#             ('qual', qual_pipeline, selected_qual_vars)
#         ])

#     # Séparation des données
#     X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

#     # Tuning des hyperparamètres avec GridSearchCV
#     st.write("### Optimisation des Hyperparamètres")

#     # Paramètres pour la régression logistique
#     logistic_param_grid = {
#         'model__C': [0.1, 1, 10],
#         'model__solver': ['liblinear', 'lbfgs']
#     }
#     logistic_model = LogisticRegression(max_iter=1000)

#     # Paramètres pour Random Forest
#     rf_param_grid = {
#         'model__n_estimators': [50, 100, 200],
#         'model__max_depth': [None, 10, 20],
#         'model__min_samples_split': [2, 5],
#         'model__min_samples_leaf': [1, 2, 4]
#     }
#     rf_model = RandomForestClassifier(random_state=42)

#     # Sélection du modèle
#     model_choice = st.selectbox("Choisissez le modèle à optimiser", ["Régression Logistique", "Random Forest"])

#     if model_choice == "Régression Logistique":
#         pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', logistic_model)])
#         grid_search = GridSearchCV(pipeline, logistic_param_grid, cv=5, scoring='accuracy')

#     elif model_choice == "Random Forest":
#         pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])
#         grid_search = GridSearchCV(pipeline, rf_param_grid, cv=5, scoring='accuracy')

#     # Entraînement et optimisation
#     st.write("Lancement de l'optimisation des hyperparamètres...")
#     grid_search.fit(X_train, y_train)

#     # Affichage des meilleurs paramètres
#     best_model = grid_search.best_estimator_
#     st.write(f"Meilleurs paramètres pour {model_choice} :")
#     st.json(grid_search.best_params_)

#     # Évaluation des performances
#     st.write("### Évaluation des Performances")
#     y_pred = best_model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_label=1)
#     recall = recall_score(y_test, y_pred, pos_label=1)
#     f1 = f1_score(y_test, y_pred, pos_label=1)
#     roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

#     # Affichage des métriques
#     metrics = {
#         'Accuracy': accuracy,
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'ROC AUC': roc_auc
#     }
#     st.write(pd.DataFrame(metrics, index=[0]))

#     # Matrice de confusion
#     conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
#     fig = go.Figure(data=go.Heatmap(
#         z=conf_matrix,
#         x=['Classe No', 'Classe Yes'],
#         y=['Classe No', 'Classe Yes'],
#         colorscale='Blues',
#         colorbar=dict(title='Nombre de prédictions')
#     ))
#     fig.update_layout(
#         title=f'Matrice de Confusion - {model_choice}',
#         xaxis_title='Prédictions',
#         yaxis_title='Vrais Labels',
#         autosize=True
#     )
#     st.plotly_chart(fig)

#     # Analyse d'équité
#     st.write("### Analyse d'Équité")
#     demographic_col = st.selectbox("Choisissez une colonne démographique pour analyser l'équité", options=qualitative_vars)

#     if demographic_col:
#         fairness_df = pd.DataFrame({
#             demographic_col: X_test[demographic_col],
#             'True': y_test,
#             'Predicted': y_pred
#         })
#         fairness_analysis = fairness_df.groupby(demographic_col).apply(
#             lambda group: pd.Series({
#                 'Accuracy': accuracy_score(group['True'], group['Predicted']),
#                 'Precision': precision_score(group['True'], group['Predicted'], pos_label=1, zero_division=0),
#                 'Recall': recall_score(group['True'], group['Predicted'], pos_label=1, zero_division=0)
#             })
#         )
#         st.write("Analyse d'équité par catégorie :")
#         st.dataframe(fairness_analysis)













elif page == pages[4]:
    st.write("### Modélisation: regression logistique  ")

    # Chargement des données
    df_Modelisation = df_Modelisation
    target = 'HadHeartAttack'  # Variable cible
    X = df_Modelisation.drop(columns=[target])
    y = df_Modelisation[target]
    
    # Conversion de la variable cible en binaire (1 pour 'Yes', 0 pour 'No')
    y = y.map({'Yes': 1, 'No': 0})

    # Sélection des variables
    quantitative_vars = X.select_dtypes(include=['number']).columns
    qualitative_vars = X.select_dtypes(exclude=['number']).columns

    selected_quant_vars = st.multiselect("Sélectionner les variables quantitatives", quantitative_vars, default=quantitative_vars)
    selected_qual_vars = st.multiselect("Sélectionner les variables qualitatives", qualitative_vars, default=qualitative_vars)

    X_selected = X[selected_quant_vars + selected_qual_vars]

    # Traitement des variables
    quant_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    qual_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('quant', quant_pipeline, selected_quant_vars),
            ('qual', qual_pipeline, selected_qual_vars)
        ])

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Tuning des hyperparamètres avec GridSearchCV
    st.write("### Optimisation des Hyperparamètres")

    # Paramètres pour la régression logistique
    logistic_param_grid = {
        'model__C': [0.1, 1, 10],
        'model__solver': ['liblinear', 'lbfgs']
    }
    logistic_model = LogisticRegression(max_iter=1000)

    # Paramètres pour Random Forest
    rf_param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2, 4]
    }
    rf_model = RandomForestClassifier(random_state=42)

    # Sélection du modèle
    model_choice = st.selectbox("Choisissez le modèle à optimiser", ["Régression Logistique", "Random Forest"])

    if model_choice == "Régression Logistique":
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', logistic_model)])
        grid_search = GridSearchCV(pipeline, logistic_param_grid, cv=5, scoring='accuracy')

    elif model_choice == "Random Forest":
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])
        grid_search = GridSearchCV(pipeline, rf_param_grid, cv=5, scoring='accuracy')

    # Entraînement et optimisation
    st.write("Lancement de l'optimisation des hyperparamètres...")
    grid_search.fit(X_train, y_train)

    # Affichage des meilleurs paramètres
    best_model = grid_search.best_estimator_
    st.write(f"Meilleurs paramètres pour {model_choice} :")
    st.json(grid_search.best_params_)

    # Évaluation des performances
    st.write("### Évaluation des Performances")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    # Affichage des métriques
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    st.write(pd.DataFrame(metrics, index=[0]))

    # Courbe ROC
    st.write("### Courbe ROC")
    y_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Tracé de la courbe ROC avec Plotly
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Courbe ROC', line=dict(color='blue', width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(color='red', dash='dash')))

    fig_roc.update_layout(
        title=f"Courbe ROC - {model_choice} (AUC = {roc_auc:.4f})",
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        xaxis=dict(scaleanchor="x", range=[0, 1]),
        yaxis=dict(scaleanchor="y", range=[0, 1]),
        autosize=True
    )

    st.plotly_chart(fig_roc)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Classe No', 'Classe Yes'],
        y=['Classe No', 'Classe Yes'],
        colorscale='Blues',
        colorbar=dict(title='Nombre de prédictions')
    ))
    fig.update_layout(
        title=f'Matrice de Confusion - {model_choice}',
        xaxis_title='Prédictions',
        yaxis_title='Vrais Labels',
        autosize=True
    )
    st.plotly_chart(fig)

    # Analyse d'équité
    st.write("### Analyse d'Équité")
    demographic_col = st.selectbox("Choisissez une colonne démographique pour analyser l'équité", options=qualitative_vars)

    if demographic_col:
        fairness_df = pd.DataFrame({
            demographic_col: X_test[demographic_col],
            'True': y_test,
            'Predicted': y_pred
        })
        fairness_analysis = fairness_df.groupby(demographic_col).apply(
            lambda group: pd.Series({
                'Accuracy': accuracy_score(group['True'], group['Predicted']),
                'Precision': precision_score(group['True'], group['Predicted'], pos_label=1, zero_division=0),
                'Recall': recall_score(group['True'], group['Predicted'], pos_label=1, zero_division=0)
            })
        )
        st.write("Analyse d'équité par catégorie :")
        st.dataframe(fairness_analysis)
