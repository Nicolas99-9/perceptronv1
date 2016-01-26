import numpy as np
import random

def data_reader(filename):
    to_binary = {"?": 3, "y": 2, "n": 1}
    labels = {"democrat": 1, "republican": -1}

    data = []
    for line in open(filename, "r"):
        line = line.strip()

        label = int(labels[line.split(",")[0]])
        observation = np.array([to_binary[obs] for obs in line.split(",")[1:]] + [1])
        data.append((label, observation))

    return data


def spam_reader(filename):
    to_binary = {1: 1, 0: -1}
    data = []
    for line in open(filename, "r"):
        line = line.strip()
        label = to_binary[int(line.split(",")[-1])]
        observation = [float(obs) for obs in line.split(",")[:-1] + [1.0]]

        data.append((label, np.array(observation)))
        
    return data

def moyenne_recursive(data,k):
    if(k==1):
        return data[0]
    m_k_1 = moyenne_recursive(data[1::],k-1)
    return (m_k_1+((data[k-1] - m_k_1)/k))

def calculate():
    n = 10 ** 6
    data = 10 ** 9 + np.random.uniform(0,1,n)
    print("moyenne : ",np.mean(data))
    print("variance : ",np.var(data)) 
    print("moyenne recusrive" , moyenne_recursive([1,2,3,4,5],len([1,2,3,4,5])))

#calculate()



#----------------------------------------------------------------- Perceptron --------------------------------------
data = data_reader("house-votes-84.data")

def getRandom(data):
    random.seed(100)
    random.shuffle(data)
    train = data[:len(data)-100]
    test = data[:100]
    return (train,test)

(train,tests) = getRandom(data) 
   

def classify(observation, poids):
    vl = np.dot(observation,poids)
    if(vl>=0):
        return 1
    return -1


def test(corpus,poids):
    erreur = 0.0
    for s in corpus:
        (value,elements) = s
        if not classify(elements,poids)== value:
            erreur +=1.0
    return erreur/len(corpus)


def learn(train,nb,poids):
    for s in range(1,nb+1): 
        for (value,elements) in train:
            if(not classify(elements,poids) == value):
                poids = poids + np.dot(elements,value)
    return poids
    
def learn_while(train,tests,nb):
    count = 0
    tmp = [0 for i in range(nb)]
    while(count<50000):
        count+=1
        tmp = learn(train,1,tmp)
        errorRate = test(tests,tmp)
        if(errorRate == 0):
            print("le progamme a converge en ",count, " taux d erreur : " , errorRate)
            return tmp
    print("le programme pas pu converger, taux d'erreur obtenu :", errorRate)
    return tmp

    
#print(test(tests,[25,-12, 67, -104, -43, 46, -18, -10, 45, -33, 54, -39, 43, -19, 5, -2, 55]))           
#mon_poids = learn(train,50)
#print(mon_poids)
#print(test(tests,mon_poids,[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))  
print(learn_while(train,tests,17))       


#---------------------------------------- Spam ---------------------------------
data2 = spam_reader("spambase.data")

def getRandom(data):
    random.seed(100)
    random.shuffle(data)
    train = data[:3600]
    test = data[:len(data)-3600]
    return (train,test)

(train2,tests2) = getRandom(data2) 
print("Nombre d erreurs pour le spam (taux en %)")
mon_poids2 = learn(train2,1,[0 for i in range(58)])
print("Valeurs des poids w du perceptron : ",mon_poids2)
print(test(tests2,mon_poids2))  
print("apres plusieurs appprentissages : ")
print(learn_while(train2,tests2,58)) 


#------------------------------------- Learn with a bias ------------------------


def learn_biais(train,nb,poids,b):
    for s in range(1,nb+1): 
        for (value,elements) in train:
            if(not classify(elements,poids) == value):
                tmp = np.dot(elements,value)
                poids = poids + np.dot(tmp,b)
    return poids
    
def learn_biais(train,tests,nb):
    count = 0
    tmp = [0 for i in range(nb)]
    while(count<50000):
        count+=1
        tmp = learn(train,1,tmp)
        errorRate = test(tests,tmp)
        if(errorRate == 0):
            print("le progamme a converge en ",count, " taux d erreur : " , errorRate)
            return tmp
    print("le programme pas pu converger, taux d'erreur obtenu :", errorRate)
    return tmp
