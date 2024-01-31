# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:58:59 2024

@author: praneeth.ponnekanti
"""

import streamlit as st
import pandas as pd
import base64
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import joblib
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertModel
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import glob
import os
import nltk
from nltk.tokenize import word_tokenize
#from transformers import XLNetTokenizer, XLNetModel

nltk.download('punkt')


#################################################################
#### Supporting Funtions
#################################################################

def to_camel_case(text):
    words = text.split()
    return ' '.join(word.capitalize() for word in words)

def loading_spinner(text):
    st.spinner(text)
        #st.text("In Progress...")s

def latest_info_joblib(directory):
    files = glob.glob(os.path.join(directory, '*_info.joblib'))
    if not files:
        return None

    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def download_table(df, op_file):
    """
    Generates a link allowing the data in a given pandas DataFrame to be downloaded as an Excel file.

    Parameters:
    - df: pandas DataFrame
    - op_file: str, the desired file name (without extension)
    - sheet_name: str, the name of the sheet in the Excel file (default: "default_sheet")

    Returns:
    None
    """
    if not op_file.endswith('.xlsx'):
        op_file += '.xlsx'
    sheet_fmt = "Generated_on_" + str(time.strftime('%d-%m-%Y'))
    output = io.BytesIO()
    # Use the BytesIO object as the filehandle
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    # Write the data frame to the BytesIO object and save it
    df.to_excel(writer, sheet_name=sheet_fmt, index=False)
    # Explicitly close the writer to ensure data is written to BytesIO
    writer.close()

    # Move the BytesIO cursor to the beginning to read its content
    output.seek(0)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data)
    payload = b64.decode()

        
    # Add JavaScript code for automatic download
    js_code = f'''var link = document.createElement('a');link.href = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{payload}";link.download = "{op_file}";link.click();'''

    # Create HTML link with the JavaScript code
    st.download_button(label=f"Click here to download {op_file}", data=payload, file_name=op_file, key="download_button")
    #html = f'<a href="javascript:void(0);" onclick="{js_code}">Click here to download the table results in an Excel file!</a>'

    #st.markdown(html, unsafe_allow_html=True)
    return


def download_table_old(df, op_file):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    #op_file = "CGS_Demand_Tracker_Processed.xlsx"
    sheet_fmt = "_" + str(time.strftime('%d-%m-%Y'))
    output = io.BytesIO()
    # Use the BytesIO object as the filehandle
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    # Write the data frame to the BytesIO object and save it
    df.to_excel(writer, sheet_name=sheet_fmt, index=False)
    # Explicitly close the writer to ensure data is written to BytesIO
    writer.close()
    
    # Move the BytesIO cursor to the beginning to read its content
    output.seek(0)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data)
    payload = b64.decode()
    
    
    html = f'<a download="{op_file}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{payload}">Click here to download the table results in an Excel file!</a>'

    st.markdown(html, unsafe_allow_html=True)
    return


###################################################################################### 
#### Core Functions & ML Pipelines
######################################################################################
def process_data(data):

    # Define columns used for filtering
    filtering_columns = ['Win/ Pipeline', "Revenue ($ '000)", 'Win Probability']

    # Display columns used for filtering
    st.write("Columns used for filtering:")
    st.write(filtering_columns)

    # Check if filtering columns are present in the data
    missing_columns = [col for col in filtering_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Error: Filtering columns not found - {', '.join(missing_columns)}")
        return None

    # Calculate number of rows before filtering
    num_rows_before = data.shape[0]

    # Apply your filtering and data manipulation operations here
    filtered_data = data[data['Win/ Pipeline'].isin(['Won', 'Pipeline', 'Unqualified'])]
    filtered_data = filtered_data[filtered_data["Revenue ($ '000)"] >= 3]
    filtered_data_pct = filtered_data[filtered_data['Win Probability'] >= 0.1]

    # Calculate number of rows after filtering
    num_rows_after = filtered_data_pct.shape[0]
    
    fil_perc = 1- (num_rows_after/num_rows_before)
    
    # Display number of rows before and after filtering
    st.write(f"Number of rows before filtering: {num_rows_before} & after filtering: {num_rows_after}")
    st.write(f"% of data filtered out : {fil_perc}")
    
    return filtered_data_pct  # Return the processed data

#-----------------------------------------------------------------------------------------------------------------------------------------------
def map_to_capability(description):
    # Capability Mapping using a dictionary of key-words associated with opportunity
    capability_dict = {
        'Intelligent Functions (SAP)': ['sap', 'ecc', 's4 hana', 's/4', 's/4 hana', 'hana', 'sd', 'sap sales',
                                        'sales and distribution', 's&d', 'o2c', 'otc', 'order to cash', 'mm', 'material management',
                                        'p2p', 'ptp', 'procure to pay', 'sap procurement', 'sap pp', 'production planning',
                                        'sap manufacturing', 'plan to produce', 'sap ibp', 'integrated business planning',
                                        'sap fi', 'sap co', 'fi/co', 'fico', 'sap finance', 'sap costing', 'record to report', 'sap qm',
                                        'sap quality management', 'sap mdg', 'master data governance', 'sap ewm',
                                        'sap warehouse management', 'sap wm', 'sap tm', 'master data', 'transportation management', 's4', 'erp'
                                        'master data','s4'],
        'Agri': ['trading', 'commodity', 'protein', 'grains', 'dairy', 'palm', 'oil seeds', 'food', 'agri',
                 'farm', 'livestock', 'meat'],
        'E2ECX': ['digital- eb2b', 'b2c', ' b2b2c', ' b2b', ' c360', ' data and analytics ', ' cdp',
                  ' 1st party data', ' digital marketing', ' e-commerce', ' digital commerce', ' d2c', ' dtc',
                  ' crm', ' loyalty', ' channel management', ' mmm(marketing mix modelling) ', 'ecommerce',
                  'marketplace', 'cx', 'platform', 'omnichannel', 'ui', 'ux', 'design','plat', 'plat.'
                  'digital transformation', 'digital'],
        'GS&T': ['revenue', 'growth', 'market entry', 'go to market', 'pricing', 'analytics', 'price pack',
                 'promotions', 'data lake', 'data strategy','data warehouse', 'data analytics', 'marketing',
                 'sales and distribution', 'data science', 'trade promotion', 'trade promotions',
                 'trade promotion optimization', 'genai', 'generative ai', 'marketing analytics', 'gen ai'
                 'sales analytics', 'launch'],
        'Intelligent Functions': ['rpa', 'robotic', 'process', 'automation', 'digital manufacturing']
    }
    tokens = word_tokenize(description.lower())
    
    for capability, keywords in capability_dict.items():
        if any(keyword in tokens for keyword in keywords):
            return capability
    
    return None  # or any default value

#---------------------------------------------------------------------------------------------------------------------------

def upload_and_process_data():
    # Functionality 1: Upload an Excel file
    uploaded_file = st.file_uploader("Upload Input Demand file. (.xlsx only)", type=["xlsx", "xls"])

    if uploaded_file:
        # Functionality 2: Display all sheet names
        sheets = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("Select Demand Sheet", sheets)

        # Functionality 3: Display data header of 5 rows
        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.write(f"Data Snapshot from '{selected_sheet}' sheet:")
        st.write(data.head(5))

        # Functionality 4: Process data
        processed_data = process_data(data)

        if processed_data is not None:
            # Functionality 3: Display processed data
            st.write("Snapshot of filtered Data:")
            st.write(processed_data.head(5))

            st.write("Mapping Key account leads.")
            st.write("3. Upload Key Account Mapping Data")
            #uploaded_mapping_file = st.file_uploader("Upload for Key Account Mapping Data", type=["xlsx", "xls"])
            selected_ka_sheet = st.selectbox("Select Key Account Mapping data.", sheets)
            if selected_ka_sheet : 
                lookup_pleads_data = pd.read_excel(uploaded_file, sheet_name=selected_ka_sheet)
                st.write(f"Data Snapshot from '{selected_sheet}' sheet:")
                st.write(data.head(5))
                loading_spinner("Key Account - Demand mapping is in progress...")
                
                to_be_added_ka_list_lower = ["cargill", "archer daniels midland company", "bunge", "louis dreyfus",
                                             "ajinomoto corporation", "barry callebaut", "danone", "brasil foods",
                                             "conagra foods inc.", "golden agri-resources limited", "tyson foods inc.",
                                             "smithfield foods, inc.", "ingham enterprises pty limited", "agropur",
                                             "schwan's food", "jbs", "u.s. foodservice", "lantmännen",
                                             "bright food (group) co. ltd.", "impossible foods inc.",
                                             "corporativo agroindustrial altex", "arla", "dole food company",
                                             "pioneer food group ltd.", "nissin foods holdings co., ltd.",
                                             "agrolimen, s.a.", "olam international limited", "the a2 milk company limited",
                                             "hormel foods corpora", "mccain foods group inc", "marfrig alimentos s/a",
                                             "wilmar international limited", "maple leaf foods", "golden state foods",
                                             "mckee foods corporation", "lamb weston", "del monte", "sysco"]
    
                existing_current_new_ka_df = lookup_pleads_data[lookup_pleads_data['*Name of Account'].str.lower().isin(to_be_added_ka_list_lower)]
                fil_master_data_clients = processed_data[' Master Client Name'].str.lower().unique().tolist()
                ka_to_added = [item for item in to_be_added_ka_list_lower if item not in fil_master_data_clients]
                camel_case_list = [to_camel_case(item) for item in ka_to_added]
    
                new_df = pd.DataFrame(camel_case_list, columns=['*Name of Account'])
                new_df['Responsible Exec.'] = 'Prince'
                new_df['Participating SM'] = 'Elan'
    
                final_ka_map = pd.concat([existing_current_new_ka_df, new_df], ignore_index=True)
                st.write ("Account - Demand Mapping is complete.")
                st.write("Snapshot of Processed Key account mapping data.")
                st.write(final_ka_map.head(5))

            #st.write("Opportunity & Capability Mapping using NLP Techniques")

            # Counting blank rows in 'Opportunity' and 'Opportunity Description'
            blank_opportunity_count = processed_data['Opportunity'].isnull().sum()
            blank_description_count = processed_data['Opportunity Description'].isnull().sum()

            st.write(f'Number of blank rows in columns Opportunity: {blank_opportunity_count} & Opportunity_description : {blank_description_count}')

            # Counting rows with both 'Opportunity' and 'Opportunity Description' as blank
            both_blank_count = processed_data[(processed_data['Opportunity'].isnull()) & (processed_data['Opportunity Description'].isnull())].shape[0]
            st.write(f'Number of rows with both Opportunity and Opportunity Description as blank: {both_blank_count}')

            algo_data = processed_data
            
            algo_data['Capability_Dictionary_Mapped'] = algo_data['Opportunity'].apply(map_to_capability)
            
            st.write ("Processed data snapshot after mapping capabilities.")
            st.write (algo_data.head(10))
            # You can continue with additional functionalities as needed
            col1,col2,col3 = st.columns(3)
            ### Edited for Github
            algo_data.to_excel("./Processed Files/CGS_Demand_Tracker_Processed.xlsx", index = False)
            if col2.button("Download results to proceed further"):
                download_table(algo_data, "CGS_Demand_Tracker_Processed.xlsx")
            return algo_data
          
#---------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------
def retrain_models(inp_data):
    # Grid 2: Retrain Models
    st.header("Section 2: Retrain Models")

    inp_data.columns = inp_data.columns.str.strip()
    
    if 'Capability' in inp_data.columns :
        inp_data['Capability'] = inp_data['Capability_Dictionary_Mapped']

    # Select only the data with filled in capability names to train the ML models.
    data = inp_data.loc[pd.notnull(inp_data['Capability']), ['Master Client Name', 'Opportunity', 'Opportunity Description', 'Capability']]
    st.code(f"Input data dimensions: {data.shape}", language="python")

    # Train-test split
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
    # Reset the index for both train_data and test_data
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    st.code(f"Training Data Shape (Rows, Columns): {train_data.shape}", language="python")
    test_prop = test_data.shape[0] / data.shape[0]
    st.code(f"Testing Data Shape (Rows, Columns): {test_data.shape}", language="python")
    st.code(f"% of data used for testing: {test_prop:.2f}", language="python")

    # Load Universal Sentence Encoder
    embed_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Load BERT-based Sentence Transformer
    embed_bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Convert text to Universal Sentence Encoder embeddings
    #st.code("Embedding using Universal Encoder in progress...")
    loading_spinner("Embedding using Universal Encoder is in progress...")
    X_train_use = embed_use(train_data['Opportunity'].astype(str) + " " + train_data['Opportunity Description'].astype(str)).numpy()
    X_test_use = embed_use(test_data['Opportunity'].astype(str) + " " + test_data['Opportunity Description'].astype(str)).numpy()
    st.info("Embedding using Universal Encoder is complete!")

    # Convert text to BERT-based Sentence Transformer embeddings
    #st.code("Embedding using BERT in progress...")
    loading_spinner("Embedding using BERT is in progress...")
    X_train_bert = embed_bert.encode(train_data['Opportunity'].astype(str) + " " + train_data['Opportunity Description'].astype(str))
    X_test_bert = embed_bert.encode(test_data['Opportunity'].astype(str) + " " + test_data['Opportunity Description'].astype(str))
    st.info("Embedding using BERT is complete !")
    
    
    st.subheader("Implementing AI/ML Techniques...")
    
    # Models and Embeddings
    classifiers = {
        #'Logistic Regression': LogisticRegression(max_iter=1000),
        #'SVM': SVC(),
        #'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        #'Neural Network': MLPClassifier(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth= None, min_samples_leaf= 2, min_samples_split= 2),
        'Neural Network': MLPClassifier(max_iter=500, activation = 'relu', alpha = 0.001, hidden_layer_sizes = (50,50)),
        'LightGBM': LGBMClassifier(),
        'Gradient Boosting Classifier' : GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,random_state=100,max_features=5 )
    }

    embeddings = {
        'Universal Sentence Encoder': (X_train_use, X_test_use),
        'BERT-based Sentence Transformer': (X_train_bert, X_test_bert),
        #'DistilBERT': (X_train_distilbert_flat, X_test_distilbert_flat)
        #'XLNet': (X_train_xlnet_flat, X_test_xlnet_flat)
        
    }

    # Dictionary to store classification reports
    classification_reports = {}

    # DataFrame to store model outputs
    model_outputs = pd.DataFrame()
    model_outputs[['Master Client Name', 'Opportunity', 'Opportunity Description', 'Actual Capability']] = test_data[
        ['Master Client Name', 'Opportunity', 'Opportunity Description', 'Capability']]

    # Iterate through models and embeddings
    for classifier_name, classifier in classifiers.items():
        for embedding_name, (X_train, X_test) in embeddings.items():
            st.write(f"Training and evaluating {classifier_name} with {embedding_name}...")

            # Train the model
            classifier.fit(X_train, train_data['Capability'])

            # Make predictions
            predictions = classifier.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(test_data['Capability'], predictions)
            report = classification_report(test_data['Capability'], predictions, zero_division=1, output_dict=True)

            # Capture the model names in the report dictionary
            model_key = f"{classifier_name}_{embedding_name}"
            classification_reports[model_key] = report

            # Save the best model based on accuracy
            best_classifier = max(classification_reports, key=lambda k: classification_reports[k]['accuracy'])

            if model_key == best_classifier:
                # Save the best model along with the embedding method
                model_info = {'model': classifier, 'embedding_method': embedding_name}
                joblib.dump(model_info, f'./Trained Models/{classifier_name.lower()}_{embedding_name.lower()}_info.joblib')
                st.info(f"Saved model using {classifier_name} with {embedding_name}.")

            # Append actual vs predicted values to the model_outputs DataFrame
            model_outputs[f"{classifier_name}_{embedding_name}_Predicted"] = predictions

            st.code(f"Accuracy ({classifier_name} with {embedding_name}): {accuracy:.2f}", language="python")
            st.code("Classification Report: ", language="python")
            st.code(report, language="python")
            st.write("\n" + "=" * 50 + "\n", language="python")  # Separating different models for better readability


    # Choose the best model based on accuracy
    best_model_name_acc = max(classification_reports, key=lambda k: classification_reports[k]['accuracy'])
    best_model_acc_info = classifiers[best_model_name_acc.split('_')[0]]
    st.info(f"The best model based on accuracy: {best_model_acc_info}")
    
    # Save the best model based on accuracy
    loading_spinner(f"Saving the best model based on accuracy ({best_model_name_acc})...")
    joblib.dump(best_model_acc_info, f'./Trained Models/best_model_{best_model_name_acc.lower()}_accuracy.joblib')
    st.info(f"Saved the best model based on accuracy ({best_model_name_acc}) !")
    
    # Save the classification reports and model outputs to the same Excel file with different sheet names
    st.write("Saving model outputs and results...")
    with pd.ExcelWriter('training_output_results.xlsx') as writer:
        pd.DataFrame.from_dict(classification_reports, orient='index').to_excel(writer, sheet_name='Classification_Reports')
        model_outputs.to_excel(writer, sheet_name='Model_Output', index=False)
    st.success("Finished saving model outputs and results.")


def predict_using_models(inp_data):
    # Grid 3: Predict Using Trained Models
    st.header("Section 3: Predict Using Trained Models")
    
    #Load the latest model
    directory_path = './Trained Models/'
    latest_model_info = latest_info_joblib(directory_path)
    #latest_model_info = latest_info_joblib()
    if latest_model_info:
        st.info(f"The latest modified .joblib file ending with '_info' is: {latest_model_info}")        
        # Load the latest saved best-performing model from training data
        loaded_model_info = joblib.load(latest_model_info)
        # Extract the loaded model and embedding method
        loaded_model = loaded_model_info['model']
        loaded_embedding_method = loaded_model_info['embedding_method'].lower()
        # Assuming unseen_data is your DataFrame with 'Opportunity' or 'Opportunity Description' columns
        # Select only the data with filled in capability names for predictions
        inp_data.columns = inp_data.columns.str.strip()
        unseen_data = inp_data.loc[pd.isnull(inp_data['Capability']) & pd.notnull(inp_data['Responsible Exec']), ['Master Client Name', 'Opportunity', 'Opportunity Description', 'Capability']]
        st.write("Unseen data dimensions:", unseen_data.shape)
        #unseen_data.columns = unseen_data.columns.str.strip()
        st.write("Loaded Model : ", loaded_model)
        st.write("Chosen Embedding : ", loaded_embedding_method)
    
        predictions_df = pd.DataFrame()
        predictions_df['Master Client Name'] = unseen_data['Master Client Name']
        unseen_data = unseen_data[['Opportunity', 'Opportunity Description', 'Capability']]
    
        # Reset the index for the unseen_data
        unseen_data.reset_index(drop=True, inplace=True)
    
        # Initialize the chosen embedding method dynamically
        if loaded_embedding_method == 'universal sentence encoder':
            embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            with st.spinner("Embedding using Universal Sentence Encoder in progress..."):
                X_unseen = embedder(unseen_data['Opportunity'].astype(str) + " " + unseen_data['Opportunity Description'].astype(str)).numpy()
            st.info("Embedding using Universal Sentence Encoder is complete !")
    
        elif loaded_embedding_method == 'bert-based sentence transformer':
            with st.spinner("Embedding using BERT in progress..."):
                embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                X_unseen = embedder.encode(unseen_data['Opportunity'].astype(str) + " " + unseen_data['Opportunity Description'].astype(str))
            st.info("Embedding using BERT complete!")
        
        # Now, you can use loaded_model to make predictions on X_unseen
        with st.spinner("Making predictions..."):
            predictions_unseen = loaded_model.predict(X_unseen)
    
        # Create a DataFrame to store the predictions
        predictions_df['Opportunity'] = unseen_data['Opportunity']
        predictions_df['Opportunity Description'] = unseen_data['Opportunity Description']
        predictions_df['Predicted_Capability'] = predictions_unseen
    
        # Save the predictions to an Excel file
        st.write("Predicted Outputs : ")
        st.write(predictions_df)
        predictions_df.to_excel('./Output Files/cgs_demand_tracker_app_predictions.xlsx', index=False)
        st.success("Predictions for unseen data saved to 'cgs_demand_tracker_app_predictions.xlsx'.")
    else:
        st.warning("No .joblib files ending with '_info' found in the directory. Please train/re-train the model.")
    return predictions_df



##########################
#### Main Function
##########################

def main():
    # Main Streamlit app
    st.title(':blue[CGS Demand Tracking Tool]')
    st.markdown(''':orange[Under Development.]''')
    st.markdown('''Contact praneeth.ponnekanti@accenture.com for more details.''')

    # Sidebar with options
    st.sidebar.markdown('''***Welcome ! This app enables the following feautres :***    
                        1. Upload demand tracker data, key account leads data.  
                        2. Map opportunities with capabilities using key-words wherever possible, else leave empty.  
                        3. Implement ML/AI Technique on the input data for blank capability names, store best performing model along with test predictions.    
                        4. Use the best performing model on unseen data to make predictions for 'Capability'.   
                        5. Downloading of the output file with predictions on unseen data.  
                        :blue[You may choose the appropriate action from the options below. ]
                        ''')
    #st.sidebar.markdown(''':blue[You may choose the appropriate action from the list below. ]''')
    #option = st.sidebar.selectbox("I would like to perform the following:", ["Upload Demand & Key-Accout Mapping data.", "Retrain Models", "Predict Using Trained Models", "Download Results"])
    option = st.sidebar.selectbox("I would like to perform the following:", ["Upload Demand & Key-Account Mapping data.", "Implement & Save the best AI/ML models to map capabilities.", "Predict & Map Capabilities to Opportunities"])
    
    if option == "Upload Demand & Key-Account Mapping data.":
        dict_mapped_data = upload_and_process_data()
    elif option == "Implement & Save the best AI/ML models to map capabilities.":
        st.warning ("Do you wish to re-train the model ? This might take time.")
        st.info("You may use the latest trained model readily available for predictions on unseen data directly too!")
        um_rt, um_ue = st.columns(2)
        if um_rt.button ("Click to train the latest processed data."):
            inp_data = pd.read_excel(r"C:\Users\praneeth.ponnekanti\Downloads\CGS_Demand_Tracker_Processed.xlsx")
            retrain_models(inp_data)
        if um_ue.button("Click to load and run the existing AI/ML model.",[True,False]) :
            inp_data = pd.read_excel(r"C:\Users\praneeth.ponnekanti\OneDrive - Accenture\cgs_demand_tracker_pr_v1.xlsx")
            predict_using_models(inp_data)
    elif option == "Predict & Map Capabilities to Opportunities":
        inp_data = pd.read_excel(r"C:\Users\praneeth.ponnekanti\Downloads\CGS_Demand_Tracker_Processed.xlsx")
        op_df = predict_using_models(inp_data)
        download_table(op_df,'cgs_demand_tracker_app.xlsx')
        

if __name__ == "__main__":
    main()
