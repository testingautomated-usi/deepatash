import tensorflow as tf
import random
import xml.etree.ElementTree as ET
import re
from random import randint, uniform

import rasterization_tools
import vectorization_tools
from config import MUTOPPROB, MUTOFPROB, MUTLOWERBOUND, MUTUPPERBOUND
from utils import  euclidean

DISTANCE = 5.0
NAMESPACE = '{http://www.w3.org/2000/svg}'

class DigitMutator:

    def __init__(self, digit):
        self.digit = digit

    def mutate(self, x_test):
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            rand_mutation_probability = random.uniform(0, 1)
            rand_mutation_prob = random.uniform(0, 1)
            if rand_mutation_probability >= MUTOPPROB:            
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 1
                else:
                    mutation = 2
            else:
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 3
                else:
                    mutation = 4

            counter_mutations += 1
            mutant_vector = self.apply_mutatation(self.digit.xml_desc, mutation, counter_mutations/20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = euclidean(self.digit.purified, rasterized_digit)

            seed_image = x_test[int(self.digit.seed)]
            xml_desc = vectorization_tools.vectorize(seed_image)
            seed = rasterization_tools.rasterize_in_memory(xml_desc)
            distance_seed = euclidean(seed, rasterized_digit)
            if distance_inputs != 0 and distance_seed <= DISTANCE and distance_seed > 0:
                condition = False


        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None



    def apply_displacement_to_mutant(self, value, extent):
        displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
        if random.uniform(0, 1) >= MUTOFPROB:
            result = float(value) + displ
        else:
            result = float(value) - displ
        return repr(result)


    def apply_mutoperator1(self, svg_path, extent):    
        # find all the vertexes
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)    
        # chose a random vertex
        num_matches = len(segments) * 2
        path = svg_path
        if num_matches > 0:  
            random_coordinate_index = randint(0, num_matches - 1)
            svg_iter = re.finditer(pattern, svg_path)
            vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
            group_index = (random_coordinate_index % 2) + 1
            value = self.apply_displacement_to_mutant(vertex.group(group_index), extent)
            path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
        else:
            print("ERROR")
            print(svg_path)
        
        return path


    def apply_mutoperator2(self, svg_path, extent):
        # find all the vertexes
        pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
        segments = pattern.findall(svg_path)
        # chose a random control point
        num_matches = len(segments) * 4
        path = svg_path
        if num_matches > 0:
            random_coordinate_index = randint(0, num_matches - 1)
            svg_iter = re.finditer(pattern, svg_path)
            control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
            group_index = (random_coordinate_index % 4) + 1
            value = self.apply_displacement_to_mutant(control_point.group(group_index), extent)
            path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
        else:
            print("ERROR")
            print(svg_path)
        return path


    def apply_mutoperator3(self, svg_path):
        # find all the vertexes
        pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
        segments = pattern.findall(svg_path)
        # chose a random control point
        num_matches = len(segments)
        path = svg_path
        if num_matches > 3:
            random_coordinate_index = randint(0, num_matches - 1)
            control_point = segments[random_coordinate_index]
            cp = "C " + control_point[0] + ',' + control_point[1] + ' ' + control_point[2] + ',' + control_point[3] + ' ' + control_point[4] + ',' + control_point[5] + ' '
            # remove a control point from path
            path = re.sub(cp,'', svg_path)
        else:
            print("ERROR")
            print(svg_path)
        return path


    def apply_mutoperator4(self, svg_path):
        # find all the vertexes
        pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
        segments = pattern.findall(svg_path)
        # chose a random control point
        num_matches = len(segments)
        path = svg_path
        if num_matches > 2:
            while (True):
                random_coordinate_index = randint(0, num_matches - 1)
                if random_coordinate_index + 1 <= num_matches -1:                
                    segment = segments[random_coordinate_index]
                    old_cp = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + segment[4] + ',' + segment[5] + ' '
                    cp = "C " + segment[0] + ',' + segment[1] + ' ' + segment[2] + ',' + segment[3] + ' ' + str(random.uniform(float(segment[0]), float(segment[4]))) + ',' + str(random.uniform(float(segment[1]), float(segment[5]))) + ' '
                    next_segment = segments[random_coordinate_index + 1]
                    new_cp = "C " + str(random.uniform(float(segment[2]), float(segment[4]))) + ',' + str(random.uniform(float(segment[3]), float(segment[5]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[0]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[1]))) + ' ' + str(random.uniform(float(segment[4]), float(next_segment[4]))) + ',' + str(random.uniform(float(segment[5]), float(next_segment[5]))) + ' '
                    path = re.sub(old_cp, cp + new_cp, svg_path)
                    break
                else:
                    continue
        else:
            print("ERROR")
            print(svg_path)
        return path


    def apply_mutatation(self, svg_desc, operator_name, mutation_extent):
        root = ET.fromstring(svg_desc)
        svg_path = root.find(NAMESPACE + 'path').get('d')
        mutant_vector = svg_path    
        if operator_name == 1:
            mutant_vector = self.apply_mutoperator1(svg_path, mutation_extent)
        elif operator_name == 2:        
            mutant_vector = self.apply_mutoperator2(svg_path, mutation_extent)  
        elif operator_name == 3:        
            mutant_vector = self.apply_mutoperator3(svg_path)
        elif operator_name == 4:        
            mutant_vector = self.apply_mutoperator4(svg_path)
        return mutant_vector