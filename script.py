# Import necessary libraries
import requests
import os
from bs4 import BeautifulSoup, Tag
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import newspaper
from newspaper import Article
import nltk
import statistics
import collections
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
import time
import openai
import pandas as pd
import re
import streamlit as st
# from apify_client import ApifyClient
import pandas as pd
import transformers
from transformers import GPT2Tokenizer
from docx import Document
import json
import base64
from io import BytesIO
# import markdown
# import html2text
from markdownify import markdownify

#openai.api_key = openai.api_key = os.environ['openai_api_key']
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('vader_lexicon')
nltk.download('inaugural')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('gutenberg')
nltk.download('genesis')
# nltk.download('trigram_collocations')
# nltk.download('quadgram_collocations')


# Define a function to scrape Google search results and create a dataframe
# from apify_client import ApifyClient
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def scrape_google(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    headings = soup.find_all("h3")
    links = []

    for heading in headings:
        try:
            link = heading.find_previous("a")['href']
        except (TypeError, KeyError):
            link = ''
        links.append(link)

    data = [(heading.get_text(), link) for heading, link in zip(headings, links)]

    df = pd.DataFrame(data, columns=["Heading", "URL"])
    # filename = f"{query}_scrap_google_result.csv"
    # df.to_csv(filename, index=False)
    # print(f"scrap google result file saved as '{filename}' ")
    return df


@st.cache_data(show_spinner=False)
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""



@st.cache_data(show_spinner=False)
def truncate_to_token_length(input_string, max_tokens=1700):
    # Tokenize the input string
    tokens = tokenizer.tokenize(input_string)
    
    # Truncate the tokens to a maximum of max_tokens
    truncated_tokens = tokens[:max_tokens]
    
    # Convert the truncated tokens back to a string
    truncated_string = tokenizer.convert_tokens_to_string(truncated_tokens)
    
    return truncated_string


# Define a function to perform NLP analysis and return a string of keyness results

@st.cache_data(show_spinner=False)
def analyze_text(text):
    # Tokenize the text and remove stop words
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('english')]
    # Get the frequency distribution of the tokens
    fdist = FreqDist(tokens)
    # Create a bigram finder and get the top 20 bigrams by keyness
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    bigrams = finder.nbest(bigram_measures.raw_freq, 20)
    # Create a string from the keyness results
    results_str = ''
    results_str += 'Top 20 Words:\n'
    for word, freq in fdist.most_common(20):
        results_str += f'{word}: {freq}\n'
    results_str += '\nTop 20 Bigrams:\n'
    for bigram in bigrams:
        results_str += f'{bigram[0]} {bigram[1]}\n'
    st.write(results_str)    
    return results_str

# Define the main function to scrape Google search results and analyze the article text

@st.cache_data(show_spinner=False)
def main(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['URL']
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    # Analyze the article text for each search result and store the keyness results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        keyness_results = analyze_text(text)
        df.at[index, 'Keyness Results'] = keyness_results
    # Return the final dataframe
    #df.to_csv("NLP_Data_On_SERP_Links_Text.csv")
    return df


# # @st.cache_data(show_spinner=False)
# def remove_classes(element):
#     if isinstance(element, Tag):
#         element.attrs = {}
#         for child in element.children:
#             remove_classes(child)

@st.cache_data(show_spinner=False)
def references(url):
    try:
        
        words_to_match = ['references', 'reference', 'sources', 'source', 'citation', 'citations']

        # Send a GET request to the URL
        response = requests.get(url)
        html_content = response.content

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the first <ul> or <ol> element after the matching words and extract the HTML
        matched_element_html = None
        for word in words_to_match:
            element = soup.find(string=lambda t: t and word in t.lower())
            if element:
                # Find the first <ul> or <ol> element after the matching words
                ul_element = element.find_next(['ul', 'ol'])
                if ul_element:
                    if ul_element.name == 'ul':
                        ul_element.name = ' '  # Replace <ul> with <ol>
                if ul_element:
                    if ul_element.name == 'ol':
                        ul_element.name = ' '  # Replace <ul> with <ol>                        

                    # Check if any <li> element within the <ul> or <ol> contains the specified text
                    if any('https://' in li.text or 'http://' in li.text or 'www' in li.text for li in ul_element.find_all('li')):
                        matched_element_html = str(ul_element)
                        break  # Stop after finding the first match

        # Remove classes from the extracted HTML content
        if matched_element_html:
            matched_element_soup = BeautifulSoup(matched_element_html, 'html.parser')
#             remove_classes(matched_element_soup)
            matched_element_html = str(matched_element_soup)

        # Convert HTML to Markdown
#         matched_element_markdown = markdownify(matched_element_html)
#         print(matched_element_markdown)
        
        # Return the scraped Markdown
        # print(matched_element_html)
        return matched_element_html


    except:
        # print("except")
        return None

# Define the main function to scrape Google search results and analyze the article text


@st.cache_data(show_spinner=False)
def analyze_serps(query):
    # Scrape Google search results and create a dataframe
    df = scrape_google(query)
    # Scrape article text for each search result and store it in the dataframe
    for index, row in df.iterrows():
        url = row['URL']
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    for index, row in df.iterrows():
        url = row['URL']
        referencess = references(url)
        df.at[index, 'Reference URLs'] = referencess
    # Analyze the article text for each search result and store the NLP results in the dataframe
    for index, row in df.iterrows():
        text = row['Article Text']
        # referencing = row['Reference URLs'] 
        # Tokenize the text and remove stop words
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('english') and 'contact' not in word.lower() and 'admin' not in word.lower()]
        # Calculate the frequency distribution of the tokens
        fdist = FreqDist(tokens)
        # Calculate the 20 most common words
        most_common = fdist.most_common(20)
        # Calculate the 20 least common words
        least_common = fdist.most_common()[-20:]
        # Calculate the 20 most common bigrams
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        bigrams = finder.nbest(bigram_measures.raw_freq, 20)
        # Calculate the 20 most common trigrams
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokens)
        trigrams = finder.nbest(trigram_measures.raw_freq, 20)
        # Calculate the 20 most common quadgrams
        quadgram_measures = QuadgramAssocMeasures()
        finder = QuadgramCollocationFinder.from_words(tokens)
        quadgrams = finder.nbest(quadgram_measures.raw_freq, 20)
        # Calculate the part-of-speech tags for the text
        pos_tags = nltk.pos_tag(tokens)
        # Store the NLP results in the dataframe
        df.at[index, 'Most Common Words'] = ', '.join([word[0] for word in most_common])
        df.at[index, 'reference urls'] = row['Reference URLs']
        Reference_final_output = ', '.join(str(url) for url in df['Reference URLs'] if url is not None)
        df.at[0, 'Final_Reference_Output'] = f"<ol>{Reference_final_output}</ol>"
        df.at[index, 'Least Common Words'] = ', '.join([word[0] for word in least_common])
        df.at[index, 'Most Common Bigrams'] = ', '.join([f'{bigram[0]} {bigram[1]}' for bigram in bigrams])
        df.at[index, 'Most Common Trigrams'] = ', '.join([f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in trigrams])
        df.at[index, 'Most Common Quadgrams'] = ', '.join([f'{quadgram[0]} {quadgram[1]} {quadgram[2]} {quadgram[3]}' for quadgram in quadgrams])
        df.at[index, 'POS Tags'] = ', '.join([f'{token}/{tag}' for token, tag in pos_tags])
        # Replace any remaining commas with spaces in the Article Text column
        df.at[index, 'Article Text'] = ' '.join(row['Article Text'].replace(',', ' ').split())
    # Save the final dataframe as an Excel file

    # print(markdownify(df.at[0, 'Final_Reference_Output']))
    # filename = f"{query}_NLP_Based_SERP_RESULT.xlsx"
    # writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # df.to_excel(writer, sheet_name='Sheet1', index=False)
    # writer._save()
    # print(f"NLP based serp result file saved as '{filename}' ")
    # Return the final dataframe
    
    file_name = f"{query}_NLP_Based_SERP_Results.csv"
    link_text = "Click here to download NLP SERP Result"
    
    st.markdown(create_download_link_csv(df, file_name, link_text), unsafe_allow_html=True)
    st.write(df)
    
    # Return the final dataframe
    return df

def create_download_link_csv(df, file_name, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href


# Define a function to summarize the NLP results from the dataframe


@st.cache_data(show_spinner=False)
def summarize_nlp(df):
    # Calculate the total number of search results
    total_results = len(df)
    # Calculate the average length of the article text
    avg_length = round(df['Article Text'].apply(len).mean(), 2)
    # Get the most common words across all search results
    all_words = ', '.join(df['Most Common Words'].sum().split(', '))
    # Get the most common bigrams across all search results
    all_bigrams = ', '.join(df['Most Common Bigrams'].sum().split(', '))
    # Get the most common trigrams across all search results
    all_trigrams = ', '.join(df['Most Common Trigrams'].sum().split(', '))
    # Get the most common quadgrams across all search results
    all_quadgrams = ', '.join(df['Most Common Quadgrams'].sum().split(', '))
    # Get the most common part-of-speech tags across all search results
    all_tags = ', '.join(df['POS Tags'].sum().split(', '))
    # Calculate the median number of words in the article text
    median_words = statistics.median(df['Article Text'].apply(lambda x: len(x.split())).tolist())
    # Calculate the frequency of each word across all search results
    word_freqs = collections.Counter(all_words.split(', '))
    # Calculate the frequency of each bigram across all search results
    bigram_freqs = collections.Counter(all_bigrams.split(', '))
    # Calculate the frequency of each trigram across all search results
    trigram_freqs = collections.Counter(all_trigrams.split(', '))
    # Calculate the frequency of each quadgram across all search results
    quadgram_freqs = collections.Counter(all_quadgrams.split(', '))
    # Calculate the top 20% of most frequent words
    top_words = ', '.join([word[0] for word in word_freqs.most_common(int(len(word_freqs) * 0.2))])
    # Calculate the top 20% of most frequent bigrams
    top_bigrams = ', '.join([bigram[0] for bigram in bigram_freqs.most_common(int(len(bigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent trigrams
    top_trigrams = ', '.join([trigram[0] for trigram in trigram_freqs.most_common(int(len(trigram_freqs) * 0.2))])
    # Calculate the top 20% of most frequent quadgrams
    top_quadgrams = ', '.join([quadgram[0] for quadgram in quadgram_freqs.most_common(int(len(quadgram_freqs) * 0.2))])
    
    #print(f'Total results: {total_results}')
    #print(f'Average article length: {avg_length} characters')
    #print(f'Median words per article: {median_words}')
    #print(f'Most common words: {top_words} ({len(word_freqs)} total words)')
    #print(f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)')
    #print(f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)')
    #print(f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)')
    #print(f'Most common part-of-speech tags: {all_tags}')
    summary = ""
    summary += f'Total results: {total_results}\n'
    summary += f'Average article length: {avg_length} characters\n'
    summary += f'Average article length: {avg_length} characters\n'
    summary += f'Median words per article: {median_words}\n'
    summary += f'Most common words: {top_words} ({len(word_freqs)} total words)\n'
    summary += f'Most common bigrams: {top_bigrams} ({len(bigram_freqs)} total bigrams)\n'
    summary += f'Most common trigrams: {top_trigrams} ({len(trigram_freqs)} total trigrams)\n'
    summary += f'Most common quadgrams: {top_quadgrams} ({len(quadgram_freqs)} total quadgrams)\n'
    #summary = '\n'.join(summary)
    #st.markdown(str(summary))
    return summary





#def save_to_file(filename, content):
    #with open(filename, 'w') as f:
        #f.write("\n".join(content))


@st.cache_data(show_spinner=False)
def generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented journalist and editor. Given the following instructions, think step by step and produce the best possible output you can."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response.strip().split('\n')

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

@st.cache_data(show_spinner=False)
def generate_content2(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented journalist and editor. Given the following instructions, think step by step and produce the best possible output you can. Return the results in Nicely formatted markdown please."},
            {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None

    
@st.cache_data(show_spinner=False)
def generate_content3(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,2500)
    #st.write(prompt)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented investigative journalist and researcher. Given the following text, please write a short paragraph providing only the most important facts and takeaways that can be used later when writing a full analysis or article."},
            {"role": "user", "content": f"Use the following text to provide the readout: {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    response = response
    return response    
    
    
    
@st.cache_data(show_spinner=False)
def generate_semantic_improvements_guide(prompt,query, model="gpt-3.5-turbo", max_tokens=2000, temperature=0.4):
    prompt = truncate_to_token_length(prompt,1500)
    #for i in range(3):
        #try:
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are an expert at Semantic SEO. In particular, you are superhuman at taking  a given NLTK report on a given text corpus compiled from the text of the linked pages returned for a google search.
            and using it to build a comprehensive set of instructions for an article writer that can be used to inform someone writing a long-form article about a given topic so that they can best fully cover the semantic SEO as shown in NLTK data from the SERP corpus. 
             Provide the result in well formatted markdown. The goal of this guide is to help the writer make sure that the content they are creating is as comprehensive to the semantic SEO with a focus on what is most imprtant from a semantic SEO perspective."""},
            {"role": "user", "content": f"Semantic SEO data for the keyword based on the content that ranks on the first page of google for the given keyword query of: {query} and it's related semantic data:  {prompt}"}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = gpt_response['choices'][0]['message']['content'].strip()
    st.header("Semantic Improvements Guide")
    st.markdown(response,unsafe_allow_html=True)
    return str(response)

        #except:
            #st.write(f"Attempt {i+1} failed, retrying...")
            #time.sleep(3)  # Wait for 3 seconds before next try

    #st.write("OpenAI is currently overloaded, please try again later.")
    #return None
    
   

@st.cache_data(show_spinner=False)
def generate_outline(topic, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Generate an incredibly thorough article outline for the topic: {topic}. Consider all possible angles and be as thorough as possible. Please use Roman Numerals for each section."
    outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("outline.txt", outline)
    return outline

@st.cache_data(show_spinner=False)
def improve_outline(outline, semantic_readout, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following article outline, please improve and extend this outline significantly as much as you can keeping in mind the SEO keywords and data being provided in our semantic seo readout. Do not include a section about semantic SEO itself, you are using the readout to better inform your creation of the outline. Try and include and extend this as much as you can. Please use Roman Numerals for each section. The goal is as thorough, clear, and useful out line as possible exploring the topic in as much depth as possible. Think step by step before answering. Please take into consideration the semantic seo readout provided here: {semantic_readout} which should help inform some of the improvements you can make, though please also consider additional improvements not included in this semantic seo readout.  Outline to improve: {outline}."
    improved_outline = generate_content(prompt, model=model, max_tokens=max_tokens)
    #save_to_file("improved_outline.txt", improved_outline)
    return improved_outline



@st.cache_data(show_spinner=False)
def generate_sections(improved_outline, model="gpt-3.5-turbo", max_tokens=2000):
    sections = []

    # Parse the outline to identify the major sections
    major_sections = []
    current_section = []
    for part in improved_outline:
        if re.match(r'^[ \t]*[#]*[ \t]*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\b', part):
            if current_section:  # not the first section
                major_sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(part)
    if current_section:  # Append the last section
        major_sections.append('\n'.join(current_section))

    # Generate content for each major section
    for i, section_outline in enumerate(major_sections):
        full_outline = "Given the full improved outline: "
        full_outline += '\n'.join(improved_outline)
        specific_section = ", and focusing specifically on the following section: "
        specific_section += section_outline
        prompt =  specific_section + ", please write a thorough section that goes in-depth, provides detail and evidence, and adds as much additional value as possible. Keep whatever hierarchy you find. Never write a conclusion part of a section unless the section itself is supposed to be a conclusion. Section text:"
        section = generate_content(prompt, model=model, max_tokens=max_tokens)
        sections.append(section)
        #save_to_file(f"section_{i+1}.txt", section)
    return sections

@st.cache_data(show_spinner=False)
def improve_section(section, i, model="gpt-3.5-turbo", max_tokens=1500):
    prompt = f"Given the following section of the article: {section}, please make thorough and improvements to this section. Keep whatever hierarchy you find. Only provide the updated section, not the text of your recommendation, just make the changes. Always provide the updated section in valid Markdown please. Updated Section with improvements:"
    prompt = str(prompt)
    improved_section = generate_content2(prompt, model=model, max_tokens=max_tokens)
    #st.markdown(improved_section)
    st.markdown(improved_section,unsafe_allow_html=True)
    return "".join(improved_section)  # join the lines into a single string

@st.cache_data(show_spinner=False)
def concatenate_files(file_names, output_file_name):
    final_draft = ''
    
    for file_name in file_names:
        with open(file_name, 'r') as file:
            final_draft += file.read() + "\n\n"  # Add two newline characters between sections

    with open(output_file_name, 'w') as output_file:
        output_file.write(final_draft)

    #print("Final draft created.\n")
    return final_draft

def wp_post(url, username, password, title, content):
    # Create a WordPress client
    client = Client(url, username, password)

    # Create a new post object
    post = WordPressPost()

    # Set the post title and content
    post.title = title
    post.content = content

    # Set the post status as 'draft'
    post.post_status = 'draft'

    # Publish the post
    client.call(NewPost(post))


@st.cache_data(show_spinner=False, experimental_allow_widgets=True) 
def generate_article(topic, model="gpt-3.5-turbo", max_tokens_outline=2000, max_tokens_section=2000, max_tokens_improve_section=4000):
    status = st.empty()
    status.text('Analyzing SERPs...')
    
    query = topic
    results = analyze_serps(query)
    summary = summarize_nlp(results)

    status.text('Generating semantic SEO readout...')
    semantic_readout = generate_semantic_improvements_guide(topic, summary,  model=model, max_tokens=max_tokens_outline)
    
    
    status.text('Generating initial outline...')
    initial_outline = generate_outline(topic, model=model, max_tokens=max_tokens_outline)

    status.text('Improving the initial outline...')
    improved_outline = improve_outline(initial_outline, semantic_readout, model=model, max_tokens=1500)
    #st.markdown(improved_outline,unsafe_allow_html=True)
    
    status.text('Generating sections based on the improved outline...')
    sections = generate_sections(improved_outline, model=model, max_tokens=max_tokens_section)

    status.text('Improving sections...')
    
    improved_sections = []
    for i, section in enumerate(sections):
        section_string = '\n'.join(section)
        status.text(f'Improving section {i+1} of {len(sections)}...')
        time.sleep(5)
        improved_sections.append(improve_section(section_string, i, model=model, max_tokens=1200))



    status.text('Finished')
    final_content = '\n'.join(improved_sections)
#     html = markdown.markdown(final_content)
#     plain_text = html2text.html2text(html)
    # Set the display option to show the complete text of a column
    pd.set_option('display.max_colwidth', None)

    

    refrencess = markdownify(results.at[0, 'Final_Reference_Output'])
    final_content = final_content + '\n' + "References" + '\n' + str(refrencess)
    #st.markdown(final_content,unsafe_allow_html=True)
    file_name = f"{query}_final_article.docx"
    link_text = "Click here to download complete article"
    st.markdown(create_download_link(final_content, file_name, link_text), unsafe_allow_html=True)
    st.markdown(final_content)
    content = final_content
    url = 'https://peblog.pivotroots.com/xmlrpc.php'
    username = 'Harshraj'
    password = "QeUei(FvTvJh&obsnN(*BUWm"
    title = "testing"
    if st.button("Publish to WordPress"):
    # Call the wp_post() function with retrieved values
        wp_post(url, username, password, title, content)

# Call the generate_article() function with input values


   
def create_download_link(string, file_name, link_text):
    # Create a new Word document
    doc = Document()
    
    # Add the string content to the document
    doc.add_paragraph(string)
    
    # Save the document to a BytesIO object
    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    
    # Encode the document as base64
    doc_base64 = base64.b64encode(doc_io.read()).decode()
    # html = markdown.markdown(final_content)
    # plain_text = html2text.html2text(html)
    
    # Create the download link
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{doc_base64}" download="{file_name}">{link_text}</a>'
    
    return href



def main():
    st.set_page_config(page_title="PharmEasy Article Generator")


    st.title('PharmEasy Article Generator')
    
    st.header("Current Features")
    st.markdown("""

* Scrapes the top 10 search results and creates SEO semantics using NLP.
* Sends the SEO semantics to GPT-3.5 to generate an outline based on the semantics.
* Improves the generated outline with the required sections.
* Uses GPT-3.5 to write the article based on the improved sections.
* After generating the article, it further improves the content and creates the final draft.
""")
    st.header("Upcoming Features")
    st.markdown("""

* References at the end of the article.
* Option to define the desired word count for the article.
* Automatic draft saving in WordPress.
* Top 5 FAQs from "People Also Ask" section.
""")

    topic = st.text_input("Enter topic:")

    # Get user input for API key
    user_api_key = st.text_input("Enter your OpenAI API key")

    if st.button('Generate Content'):
        if user_api_key:
            openai.api_key = user_api_key
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
                #st.markdown(final_draft)
        else:
            st.warning("Please enter your OpenAI API key above.")

if __name__ == "__main__":
    main()





