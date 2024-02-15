#STILL UNDERDOING
from mainFinal import *
pos_query = input('Enter your Query for the Positional Index:') #"caeser OR mercy"

########################################################################

########################################################################
def pos_query_search(q, display=1):
    lis = [[] for _ in range(10)]
    ps = PorterStemmer()
    q = [ps.stem(word) for word in q.split()]

    for term in q:
        if term in invertedIndex:
            for doc_id, position in invertedIndex[term]:
                if not lis[doc_id - 1] or lis[doc_id - 1][-1] == position - 1:
                    lis[doc_id - 1].append(position)

    prefix = 'document ' if display == 1 else 'doc'
    positions = [f'{prefix}{doc_id}' for doc_id, positions_list in enumerate(lis, start=1) if len(positions_list) == len(q)]

    return positions
###########################################################################################
def do_or(z):
    subLeft = ' '.join(z.split()[:have_or(z.split())])
    subRight = ' '.join(z.split()[have_or(z.split()) + 1:])
    finalList=[]
    finalList=pos_query_search(subLeft)
    finalList.extend(x for x in pos_query_search(subRight) if x not in pos_query_search(subLeft))
    return finalList
##########################################################################################
pos_query_words = pos_query.split()
def have_and(x):
    return x.index('AND') if 'AND' in x else -1
def have_or(x):
    return x.index('OR') if 'OR' in x else -1
def have_not(x):
    return x.index('NOT') if 'NOT' in x else -1

if have_and(pos_query_words) != -1:
    y = ' '.join(pos_query_words[:have_and(pos_query_words)])
    z = ' '.join(pos_query_words[have_and(pos_query_words) + 1:])
    if have_or(y) != -1 :
        print(list(set(do_or(y)) & set(pos_query_search(z))))  
                
    elif have_or(z) != -1:
        print(list(set(do_or(z)) & set(pos_query_search(y))))
          
    else:
        print(list(set(pos_query_search(y)) & set(pos_query_search(z))))
    
elif have_or(pos_query_words) != -1:
    print(do_or(pos_query))
    
elif have_not(pos_query_words) != -1:
    y = ' '.join(pos_query_words[:have_not(pos_query_words)])
    z = ' '.join(pos_query_words[have_not(pos_query_words) + 1:])
    print(list(set(pos_query_search(y)) - set(pos_query_search(z))))
else:
    print(pos_query_search(pos_query))
