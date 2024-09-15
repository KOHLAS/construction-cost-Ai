import streamlit as st
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import json
import re
import openai

# Function to convert markdown table to CSV
def markdown_table_to_csv(markdown_table):
    rows = markdown_table.strip().split('\n')
    columns = [col.strip() for col in rows[0].split('|')[1:-1]]
    data = []
    for row in rows[2:]:
        data.append([cell.strip() for cell in row.split('|')[1:-1]])
    df = pd.DataFrame(data, columns=columns)
    return df

# Function to calculate costs based on user input
def calculate_costs(df, cost_dict):
    df['Cost per Unit'] = df['Object Name'].map(cost_dict)
    df['Total Cost'] = df['Cost per Unit'].astype(float) * df['Quantity'].astype(float)
    return df

st.set_page_config(page_title='Blueprint Take-off AI', page_icon='👁️')

st.markdown('# CAD Blueprint Take-off AI')
api_key = st.text_input('OpenAI API Key', '', type='password')

# Get user inputs
img_input = st.file_uploader('Upload Blueprint Images', accept_multiple_files=True)
cost_input = st.text_area('Input Costs (e.g., Concrete:100, Steel:50)', '')

# Convert cost input to a dictionary
if cost_input:
    cost_dict = dict(re.findall(r'(\w+):(\d+)', cost_input))

# Send API request to OpenAI
if st.button('Send'):
    if not api_key:
        st.warning('API Key required')
        st.stop()

    # Prepare API request
    msg = {'role': 'user', 'content': []}
    msg['content'].append({
        'type': 'text', 
        'text': 'Provide a take-off of the quantities from this engineering drawing, returning ONLY as a markdown table.'
    })

    images = []
    for img in img_input:
        if img.name.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            st.warning('Only .jpg, .png, .gif, or .webp are supported')
            st.stop()
        encoded_img = base64.b64encode(img.read()).decode('utf-8')
        images.append(img)
        msg['content'].append(
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{encoded_img}',
                    'detail': 'low'
                }
            }
        )

    # Initialize OpenAI client
    openai.api_key = api_key
    try:
        # Send request to GPT model
        response = openai.ChatCompletion.create(
            model='gpt-4',
            temperature=0.0,
            max_tokens=300,
            messages=[msg]
        )
        response_msg = str(response.choices[0].message['content'])
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI Error: {e}")
        st.stop()

    # Display user input and response
    with st.chat_message('user'):
        for i in msg['content']:
            if i['type'] == 'text':
                st.write(i['text'])
            else:
                with st.expander('Attached Image'):
                    img = Image.open(BytesIO(base64.b64decode(i['image_url']['url'][23:])))
                    st.image(img)
    
    # Process GPT response
    if response_msg:
        with st.chat_message('assistant'):
            st.markdown(response_msg)
            # Assume the entire response_msg is the markdown table
            try:
                df = markdown_table_to_csv(response_msg)
                if not df.empty and cost_input:
                    # Calculate costs if cost input is provided
                    df = calculate_costs(df, cost_dict)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download table as CSV",
                    data=csv,
                    file_name='table_with_costs.csv' if cost_input else 'table.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error processing table: {e}")
