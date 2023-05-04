import random
from utils import edit_distance
from config import DISTANCE
from utils import find_adjs, listToString, get_synonym, untokenize
import logging as log
import Levenshtein as lev

class TextMutator:


    def __init__(self, text):
        self.text = text

    def mutate(self, x_test):
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            mutation = random.choice([1,2,3])

            counter_mutations += 1

            mutant_vector = self.apply_mutatation(self.text.text, mutation)

            distance_inputs = lev.distance(self.text.text, mutant_vector)
            distance_seeds = lev.distance(x_test[self.text.seed], mutant_vector)
            edit_distance_seeds = edit_distance(x_test[self.text.seed], mutant_vector)
            if distance_inputs != 0 and edit_distance_seeds <= DISTANCE and distance_seeds != 0:
                condition = False


        self.text.text = mutant_vector
        self.text.predicted_label = None

    # replace a word with its synonym
    def apply_mutoperator1(self, text):    

        # print("*************** replace a word with its synonym")

        vector = text.split()
        syn = None

        count = 0
        while syn == None or syn == []:
            rand_index = random.randint(0, len(vector)-1)  
            syn = get_synonym(vector[rand_index])
            count += 1
            if count > 2000:
                log.info("no synonym found!")
                break

        if syn != None and syn != []:
            vector[rand_index] = syn 
            
            text = untokenize(vector)


        return text

    # duplicate a sentence
    def apply_mutoperator2(self, text):

        # print("*************** duplicate a sentence")

        vector = text.split('.')
        rand_index = random.randint(0, len(vector)-1)   

        text = text + " " + vector[rand_index] + "."
        return text


    # add synonym
    def apply_mutoperator3(self, text):    

        # print("*************** add a synonym")

        vector, adjs_index = find_adjs(text)
        final_vector = []
        syn = None

        # check if we have any adj in sentence
        if len(adjs_index) > 0:
            count = 0
            while syn == None or syn == []:
                rand_index = random.choice(adjs_index)  
                syn = get_synonym(vector[rand_index][0])
                count += 1
                if count > 2000:
                    log.info("no synonym found!")
                    break

            if syn != None and syn != []:
                vector[rand_index] = (vector[rand_index][0] + " and " + syn, vector[rand_index][1])

                # traverse in the string  
                for ele in vector: 
                    final_vector.append(ele[0])

                text = untokenize(final_vector)
        else:
            log.info("no adjective found!")

        return text


    def apply_mutatation(self, text, operator_name):
        mutant_vector = text    
        if operator_name == 1:
            mutant_vector = self.apply_mutoperator1(text)
        elif operator_name == 2:        
            mutant_vector = self.apply_mutoperator2(text)
        elif operator_name == 3:
            mutant_vector = self.apply_mutoperator3(text)
        return mutant_vector