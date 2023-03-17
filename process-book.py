# Import libraries
import PyPDF2
import json
import re
import os
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
import openai
import time
import numpy as np
from sklearn.cluster import KMeans

# openai.api_key      = os.environ['OPENAI_API_KEY']
openai.api_key      = input('Enter OpenAI API key: ')

book_name           = input('Enter book name: ')

# Make directory for book if it doesn't exist
if not os.path.exists(f'BookProcessed/{book_name}'):
    os.mkdir(f'BookProcessed/{book_name}')

book_path           = f'BookData/PDF/{book_name}.pdf'
text_path           = f'BookProcessed/{book_name}/{book_name}.json'
embeddings_path     = f'BookProcessed/{book_name}/embeddings.csv'
clusters_path       = f'BookProcessed/{book_name}/clusters.csv'
summary_path        = f'BookProcessed/{book_name}/summary.csv'
quotes_path         = f'BookProcessed/{book_name}/quotes.csv'
overview_path       = f'BookProcessed/{book_name}/overview.txt'
practical_path      = f'BookProcessed/{book_name}/practical_applications.csv'

# {chapterNumber: [chapterTitle, chapterBeginPageNumber]}

# read json file
chapters            = {}
with open(f'BookData/ChapterJson/{book_name}.json') as json_file:
    chapters = json.load(json_file)

# Utility function to return response from ChatGPT
def get_response(user_prompt, system_prompt="You are a helpful AI assistant"):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print("An error occurred while generating response: " + str(e))
            time.sleep(3)
            continue

# Function to generate text from the book PDF
def generate_book_text():

    print('Generating book text...')

    with open(book_path, 'rb') as pdf_file:
        
        pdf_reader          = PyPDF2.PdfReader(pdf_file)
        num_pages           = len(pdf_reader.pages)

        all_chapters        = {}
        curr_chapter        = 'chapter 1'
        curr_chapter_text   = []


        chapter_begin_pages = [chapter_number[1] for chapter_number in chapters.values()]

        for page_num in range(num_pages):

            page_obj = pdf_reader.pages[page_num]

            # Convert all whitespace to single spaces
            page_text = re.sub(r'\s+', ' ', page_obj.extract_text())

            # Reset current chapter data if new chapter found
            if page_num in chapter_begin_pages:

                curr_chapter = 'chapter {}'.format(chapter_begin_pages.index(page_num) + 1)
                curr_chapter_text = []

            if curr_chapter not in all_chapters:
                all_chapters[curr_chapter] =    {
                                                    'name': chapters[curr_chapter][0], 
                                                    'text': []
                                                }

            # Split the page text into chunks of 200 words and append to current chapter text
            words = page_text.split()
            for i in range(0, len(words), 200):
                curr_chapter_text.append(' '.join(words[i:i+200]))

            # Add current chapter name and text to all chapters dictionary
            all_chapters[curr_chapter] =    {
                                                'name': chapters[curr_chapter][0], 
                                                'text': curr_chapter_text
                                            }

    pdf_file.close()

    with open(text_path, 'w') as file:
        json.dump(all_chapters, file)

    print('Saved book text as JSON file')

# Function to generate embeddings for each chunk of text
def generate_book_embeddings():

    print('Generating book embeddings...')

    f                      = open(text_path, 'r')
    data                   = json.load(f)

    df                     = pd.DataFrame()

    pd_data                 =   {
                                'Chapter Name' : [], 
                                'Chunk Number' : [], 
                                'Content' : [], 
                                'Content Length' : [], 
                                'Token Length' : [], 
                                'Embedding Vector' : []
                            }

    embedding_model         = "text-embedding-ada-002"
    embedding_encoding      = "cl100k_base"

    encoding                = tiktoken.get_encoding(embedding_encoding)

    for _, chapter_data in data.items():

        chapter_text        = chapter_data['text']
        chapter_name        = chapter_data['name']

        for i, chunk in enumerate(chapter_text):

            start_time = time.time()

            content = get_response(user_prompt="Clean the following paragraphs and make it readable without changing the content. Only correct spelling and punctuation: " + chunk)
            
            pd_data['Chapter Name'].append(chapter_name)
            pd_data['Chunk Number'].append(i)
            pd_data['Content'].append(content)
            pd_data['Content Length'].append(len(content))
            pd_data['Token Length'].append(len(encoding.encode(content)))
            pd_data['Embedding Vector'].append(get_embedding(content, embedding_model))

            time_taken = time.time() - start_time
            print('Performed cleaning of chunk', (i + 1), '/', len(chapter_text), 'in', chapter_name, 'in', time_taken, 'seconds')

            # Save embeddings after every 10 chunks
            if (i + 1) % 10 == 0:
                    
                    df = pd.DataFrame(pd_data)
                    df.to_csv(embeddings_path, index=False)
        
        df = pd.DataFrame(pd_data)
        df.to_csv(embeddings_path, index=False)
    
    print('Saved embeddings for all chapters')


# Generate the book clusters
def generate_book_clusters():

    print('Generating book clusters...')

    df                      = pd.read_csv(embeddings_path)
    df["Embedding Vector"]  = df['Embedding Vector'].apply(eval).apply(np.array)

    cluster_summary = pd.DataFrame()

    pd_data                 =   {
                                    'Chapter Name' : [], 
                                    'Chapter Summary' : []
                                }

    chapters                = df['Chapter Name'].unique()

    chapter_no = 1
    for chapter in chapters:

        # Get embeddings for current chapter
        chapter_data = df[df['Chapter Name'] == chapter].copy()
        matrix = np.vstack(chapter_data['Embedding Vector'].values)

        n_clusters = min(8, len(chapter_data))
        kmeans = KMeans(
                            n_clusters = n_clusters, 
                            init = "k-means++", 
                            random_state = 42, 
                            n_init = 'auto'
                        )
        
        # Find n_cluster clusters for current chapter
        kmeans.fit(matrix)

        # Add a cluster label to chapter data
        labels = kmeans.labels_
        chapter_data["Cluster"] = labels
        
        cluster_summary = []

        # Get at most 3 chunks from each cluster and generate a summary for that cluster
        # The idea is to summarise the concepts in that cluster
        for i in range(n_clusters):

            print('In cluster: ', i, '/', n_clusters, ',of chapter: ', chapter_no, ' of ', len(chapters), ' chapters.')

            number_of_chunks = min(3, np.bincount(kmeans.labels_)[i])
            chunks = chapter_data.loc[chapter_data['Cluster'] == i, 'Content'].sample(number_of_chunks, random_state=42)
            chunks = ' '.join(chunks.values)

            response = get_response(
                system_prompt = f"You are the most well read agent on the book {book_name}",
                user_prompt = "You are tasked with creating an amazing chapter summary of chapters in the book. To assist you in this task, I will give you paragraphs from the book that are similar in nature and also the chapter name. I need you to condense all that information into about 100 words. This information should be the important takeaways which could include things like ideas and concepts. It should, however, not include things such as long drawn explanations, descriptions or anything irrelevant. Your tone should be fun, helpful, witty and likeable. Do not sound like a sage. Write in a way that draws people to read more." + "\n\nChapter " + str(chapter_no) + ": " + chapter + "\n\nParagraphs:\n\n" + chunks
            )

            cluster_summary.append(response)

        # Save the cluster summary for the chapter
        pd_data['Chapter Name'].append(chapter)
        pd_data['Chapter Summary'].append(cluster_summary)
        chapterwise_clusters = pd.DataFrame(pd_data)
        chapterwise_clusters.to_csv(clusters_path, index=False)
        print('Done with chapter clusters of chapter: ', chapter_no, ' of ', len(chapters), ' chapters.')
        chapter_no += 1


# Generate the chapter summaries
def generate_chapter_summaries():

    print('Generating chapter summaries...')

    df = pd.read_csv(clusters_path)

    chapter_clusters                = df['Chapter Summary'].values
    chapter_names                   = df['Chapter Name'].values

    summaries                       =   {
                                            'Summaries': []
                                        }

    # Iterate over every chapter and its cluster summary
    for cluster in chapter_clusters:

        chapter_cluster = eval(cluster)
        combined_summary = ' '.join(chapter_cluster)

        # Generate a 5 point summary for the chapter
        response = get_response(
            user_prompt = f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing chapter summary of this book. To assist you with this task, I will give you the name of the chapter and a comprehensive summary of the chapter which is about 800 words. 

                                Your task: Write a 5 point summary from my input paragraph. These points should capture the most important and useful ideas and concepts from the chapter. Your writing style and tone should be fun, lively, productive, helpful, witty and likeable but do not sound like a sage. Each point should be about 80 words and must have a point summary. Your output format should be a json file with the following structure:
                                                                    
                                {{
                                    "<Point 1 Title (Summary)>": "<Point 1 Content>",
                                    "<Point 2 Title (Summary)>": "<Point 2 Content>",
                                    "<Point 3 Title (Summary)>": "<Point 3 Content>",
                                    "<Point 4 Title (Summary)>": "<Point 4 Content>",
                                    "<Point 5 Title (Summary)>": "<Point 5 Content>"
                                }}

                                Chapter: {chapter_names[len(summaries['Summaries'])]}

                                Paragraphs: {combined_summary}
                            """
        )

        summaries['Summaries'].append(response)
        print('Done with chapter summary of chapter: ', len(summaries['Summaries']), ' of ', len(chapter_clusters), ' chapters.')

        final_summary = pd.DataFrame(summaries)
        final_summary.to_csv(summary_path, index=False)

# Generate book quotes
def generate_book_quotes():

    print('Generating book quotes...')

    df                              = pd.read_csv(clusters_path)

    chapter_clusters                = df['Chapter Summary'].values
    chapter_names                   = df['Chapter Name'].values

    quotes                          =   {
                                            'Quotes': []
                                        }

    # Generate the top 2 quotes from each chapter
    for cluster in chapter_clusters:

        chapter_cluster = eval(cluster)
        combined_summary = ' '.join(chapter_cluster)

        response = get_response(
            user_prompt =   f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing top quotes of this book. To assist you with this task, I will give you the name of the chapter and a comprehensive summary of the chapter which is about 800 words. 

                                Your task: Output the top 2 quotes from the chapter summary. The output should be in the format:

                                {{
                                    "quote 1" : "<Quote 1>"
                                    "quote 2" : "<Quote 2>"
                                }}

                                Chapter: {chapter_names[len(quotes['Quotes'])]}

                                Paragraphs: {combined_summary}
                            """
        )
        quotes['Quotes'].append(response)
    
    quotes = pd.DataFrame(quotes)

    print('Generated first set of quotes for every chapter')

    chapter_quotes                  = quotes['Quotes'].values

    final_quotes                    =   {
                                            'Quotes': []
                                        }

    # Iterate through chapter wise quotes 5 chapters at a time
    # Then pick the top 2 quotes from all the 5 chapters
    for i in range(0, len(chapter_quotes), 5):

        combined_quote_summary = ' '.join(chapter_quotes[i:i+5])

        response = get_response(
            user_prompt =   f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing top quotes of this book. To assist you with this task, I will give you a couple of quotes from the book. 

                                Your task: Output the top 2 quotes from the quotes I give you. The output should be in the format:

                                {{
                                    "quote 1" : "<Quote 1>"
                                    "quote 2" : "<Quote 2>"
                                }}

                                Quotes: {combined_quote_summary}
                            """
        )

        final_quotes['Quotes'].append(response)

    print('Generated final set of quotes for the book')

    final_quotes = pd.DataFrame(final_quotes)
    final_quotes.to_csv(quotes_path, index=False)        


# Generate Book overview
def generate_book_overview():

    print('Generating book overview...')
    
    df                                  = pd.read_csv(summary_path)

    summaries                           = df['Summaries'].values

    overview_points                     =   {
                                                'Overvew': []
                                            }

    # First send chunks of 5 chapter summaries at a time to create a 150 word summary
    for i in range(0, len(summaries), 5):

        combined_summary = ' '.join(summaries[i:i+5])

        response = get_response(
            user_prompt =   f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing summary of this book. To assist you with this task, I will give you some points for summary.

                                Your task: Condense all this information in summary form in about 150 word. Your output must be strictly related to the points I give you. 
                                
                                Input: {combined_summary}
                            """
        )
        overview_points['Overvew'].append(response)

    # Send a final request to condense 5-chapter-overviews into a 100 word summary
    overview = get_response(
        user_prompt =   f"""
                            You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing summary of this book. To assist you with this task, I will give you some points for summary.
                
                            Your task: Condense all this information in summary form in about 100 words. Your output must be strictly related to the points I give you. 
                            
                            Input: {' '.join(overview_points['Overvew'])}
                        """
    )

    with open(overview_path, 'w') as f:
        f.write(overview)

# Generate practical applications
def generate_book_practical_applications():

    print('Generating book practical applications...')

    df = pd.read_csv(clusters_path)

    chapter_clusters                    = df['Chapter Summary'].values
    chapter_names                       = df['Chapter Name'].values

    examples                            =   {
                                                'Summaries': []
                                            }

    # First generate practical applications for each chapter using 800 word chapter clusters
    for cluster in chapter_clusters:

        chapter_cluster = eval(cluster)
        combined_summary = ' '.join(chapter_cluster)

        response = get_response(
            user_prompt =   f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing practical application points of this book. To assist you with this task, I will give you the name of the chapter and a comprehensive summary of the chapter which is about 800 words. 

                                Your task: Write 2 practical examples that a reader should implement in their daily life which is closely related to the chapter summary. The point should be implementable and simple. Your writing style and tone should be assertive, for instance instead of "could", say "do". Each point should be about 50 words and must have a point summary. Your output format should be:

                                {{
                                    "<Point 1 Summary>" : "<Point 1 Content>"
                                    "<Point 2 Summary>" : "<Point 2 Content>"
                                }}

                                Chapter: {chapter_names[len(examples['Summaries'])]}
                                
                                Paragraphs: {combined_summary}
                            """
        )

        examples['Summaries'].append(response)


    examples                            = pd.DataFrame(examples)

    practical_applications              = examples['Summaries'].values

    all_examples                        =   {
                                                'Examples': []
                                            }
    
    print('Generated first set of summaries for practical examples')

    #Iterate over previously generated examples 5 at a time and summarise them again
    for i in range(0, len(practical_applications), 5):

        practical_examples_5 = practical_applications[i:i+5]
        combined_example_summary = ' '.join(practical_examples_5)

        response = get_response(
            user_prompt =   f"""
                                You are the most well read agent on the book {book_name}. You are tasked with creating the most amazing practical application points of this book. To assist you with this task, I will give you several practical examples. 

                                Your task: Tell me the top 2 practical examples (from the ones I gave you) that a reader should implement in their daily life. The point should be implementable and simple. Your writing style and tone should be assertive, for instance instead of "could", say "do". Each point should be about 50 words and must have a point summary. Your output format should be:

                                {{  
                                    "<Point 1 Summary>" : "<Point 1 Content>"
                                    "<Point 2 Summary>" : "<Point 2 Content>"
                                }}

                                Paragraphs: {combined_example_summary}
                            """
        )

        all_examples['Examples'].append(response)

        final_summary = pd.DataFrame(all_examples)
        final_summary.to_csv(practical_path, index=False)

    print('Generated second set of summaries for practical examples')



start_time = time.time()
generate_book_text()
print(f'Book text generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_book_embeddings()
print(f'Book embeddings generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_book_clusters()
print(f'Book clusters generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_chapter_summaries()
print(f'Chapter summaries generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_book_quotes()
print(f'Book quotes generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_book_overview()
print(f'Book overview generated in {time.time() - start_time} seconds')
start_time = time.time()
generate_book_practical_applications()
print(f'Book practical applications generated in {time.time() - start_time} seconds')