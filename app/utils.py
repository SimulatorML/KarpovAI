from typing import List, Tuple, Any
from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem
import numpy as np
import nltk
nltk.download('stopwords')
mystem = Mystem()
rus_stopwords = stopwords.words('russian')


async def preprocess_text(text: str) -> List:
    """Tokenizes text"""
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in rus_stopwords
              and token != " "
              and token.strip() not in punctuation
              and token != 'https'
              and token != '://'
              and not token.isdigit()]
    return tokens


async def predict_with_trained_model(message: str,
                                     bm25_desc: Any,
                                     bm25_title: Any,
                                     links: Any) -> List[Tuple[str, str]]:
    """
    Recommends videos relevant to the user's query
    Parameters
    ----------
    message: str
      User message
    bm25_desc: pickle
      The model trained according to descriptions in pkl format
    bm25_title: pickle
      The model trained according to titles in pkl format
    links: json
      Downloaded json file with links and titles

    Returns
    -------
    List[Tuple[str, str]]
      List with tuples containing link and title
    """
    message = await preprocess_text(message)

    scores_desc = bm25_desc.get_scores(message)
    index_desc = list(map(str, np.argsort(scores_desc)[-3:]))

    scores_title = bm25_title.get_scores(message)
    index_title = list(map(str, np.argsort(scores_title)[-3:]))


    index = index_desc + index_title

    score = []
    for ind in index:
        score.append((max(scores_desc[int(ind)], scores_title[int(ind)]), ind))
    score = sorted(list(set(score)), key=lambda tup: tup[0], reverse=True)[:3]

    #обрезаем где скор слишком маленький
    score = [item for item in score if item[0] > 6.6]

    result = []
    if len(score):
        for scr in score:
            result.append((links[scr[1]]["link"], links[scr[1]]["title"]))

    return result
