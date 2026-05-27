import pandas as pd
import numpy as np
import multiprocessing as mp
import tempfile, shutil, gc, os
from sentence_transformers import SentenceTransformer
from typing import Literal, Optional
import torch

class BGEEmbedder:
    '''
    Embeddings Class Package that has three features. 
        - Can accomodate large or small embeddings model.
        - Can implement parallel coding into its algorithm. 
        - For larger datasets, has an option to clear the cache to save up on RAM. 
    
    Usage:
        - Typical usage should be just calling the class, specifying the number of cores (if you want parallel processing for larger tasks), and then after, specifying a save path for the embedder to save a file to.  
        - It will return a new data frame with an extra column called "embedding" which specifies the direction/vector each column is associated with. 
    '''



    def __init__(self, model_size: Literal["large", "small"] = "large", n_cores: Optional[int] = None):
        self.model_size = model_size
        self.n_cores = n_cores or max(1, mp.cpu_count() - 1)
        self.model_name = f"BAAI/bge-{model_size}-en-v1.5"
    
    def embed(self, df: pd.DataFrame, text_col: str = "Text", chunk_size: int = None, 
              batch_size: int = 32, save_path: str = None) -> pd.DataFrame:
        
        '''
        Arguments:
        Required Arguments:
        text column: Name of the text column we will be creating embeddings from! Make sure to check that it has proper context. 
        df: The dataframe that we are trying to embed. 
        
        Semi Required:
        save_path: Path that we will save the embeddings to. Should be a .npy file. If we want to save embeddings then, we should specify this.
        Batch Size: Batch size determines how many rows we are processing at a given moment. (Low for less RAM, High for higher RAM)
                - If we have a batch size of 32, we process 32 rows initially. 
        Chunk Size: Will control how we divide up the data frame initially. 
                - If there are four chunks and we have 10,000 rows, will give us four chunks of (1-2500), (2501-5000), (5001-7500), (7501-10000)


        Returns: A Data frame with a new column called "embedding" 

        '''
        
        texts = df[text_col].fillna("").tolist()
        
        if chunk_size:
            embeddings = self._chunked_parallel_embed(texts, chunk_size, batch_size)
        else:
            model = SentenceTransformer(self.model_name)  
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True) 
            del model                           # Deletes the model immediately after finishing. 
            gc.collect()                        # Cleans up the extra ram stored. So if initializing arrays or creating other models, will delete those initially. 
        
        if save_path:
            np.save(save_path, embeddings)      # saves embedding to directed path of choice. 
        
        df["embedding"] = embeddings.tolist()   # Turns all the embeddings into a list and saves the final result to our original data frame. 
        return df                               # Final step returning all of the data. 
    
    def _chunked_parallel_embed(self, texts, chunk_size, batch_size):
        n_cores = self.n_cores
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        temp_dir = tempfile.mkdtemp()
        
        '''
        No required arguments from here. This function is only used to help linearly program. 
        '''

        print(f"Embedding {len(texts)} texts | {len(chunks)} chunks | {n_cores} cores")
        
        try:
            for chunk_id, chunk in enumerate(chunks):
                print(f"  Processing chunk {chunk_id + 1}/{len(chunks)}...")
                
                sub_chunks = self._split_into_n(chunk, n_cores)
                args = [(sub, self.model_name, batch_size) for sub in sub_chunks]
                
                with mp.Pool(processes=n_cores) as pool:
                    sub_results = pool.map(self._embed_sub_chunk, args)
                
                chunk_embeddings = np.vstack(sub_results)
                np.save(f"{temp_dir}/chunk_{chunk_id}.npy", chunk_embeddings)
                
                del chunk_embeddings, sub_results, sub_chunks, chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_embeddings = np.vstack([np.load(f"{temp_dir}/chunk_{i}.npy") for i in range(len(chunks))])
            return all_embeddings
        
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def _embed_sub_chunk(args): 
        
        '''
        THIS IS NEEDED FOR MULTI-PROCESSING. 
        AFTER EACH CHUNK, DELETES AND FREES UP MEMORY AND CREATES ALTERNATIVE MODELS. 
        '''
        texts, model_name, batch_size = args
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings
    
    @staticmethod
    def _split_into_n(items, n): # Splits number of items into n supposed items. 
        k = max(1, len(items) // n)
        return [items[i:i + k] for i in range(0, len(items), k)]
    