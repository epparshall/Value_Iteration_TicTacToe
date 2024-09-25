import numpy as np
import matplotlib.pyplot as plt 
import time

class Value_Iteration_TicTacToe():
    def __init__(self, gamma=.89, start_V=None):
        self.gamma = gamma
        self.possible_states = 19683
        self.states_matrix = self.create_all_states()

        if (start_V is None):
            self.V = np.zeros((self.possible_states, 1))
            self.new_V = np.zeros((self.possible_states, 1))
        else:
            self.V = start_V
            self.new_V = np.zeros((self.possible_states, 1))
        
        self.valid_states = self.get_valid_states()

    def print_board(self, state):
        def convert_num_to_letter(num):
            if (num == 0):
                return " "
            if (num == 1):
                return "X"
            if (num == 2):
                return "O"
            raise("Give me a real number bro")
        
        # 0 = Blank, 1 = X, 2 = 0

        print()
        print("   " + str(convert_num_to_letter(state[0])) + " | " + str(convert_num_to_letter(state[1])) + " | " + str(convert_num_to_letter(state[2])))
        print("----------------")
        print("   " + str(convert_num_to_letter(state[3])) + " | " + str(convert_num_to_letter(state[4])) + " | " + str(convert_num_to_letter(state[5])))
        print("----------------")
        print("   " + str(convert_num_to_letter(state[6])) + " | " + str(convert_num_to_letter(state[7])) + " | " + str(convert_num_to_letter(state[8])))
        print()  

    def create_all_states(self):
        states_matrix = np.zeros((9, self.possible_states), dtype=int)  

        for i in range(self.possible_states):
            base3_num = np.base_repr(i, base=3, padding=0)
            res = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            for j in range(len(str(base3_num))):
                res[-j-1] = int(str(base3_num[-j-1]))

            states_matrix[:, i] = res
        
        return states_matrix
    
    def evaluate_reward(self, state):
        # We want X to win so winning state for X is reward of 1 and winning state for O is -1

        #  0 | 1 | 2
        # ----------------
        #  3 | 4 | 5
        # ----------------
        #  6 | 7 | 8

        reward = 0

        if ((state[0] == 1) and (state[1] == 1) and (state[2] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[3] == 1) and (state[4] == 1) and (state[5] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[6] == 1) and (state[7] == 1) and (state[8] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[0] == 1) and (state[3] == 1) and (state[6] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[1] == 1) and (state[4] == 1) and (state[7] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[2] == 1) and (state[5] == 1) and (state[8] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[0] == 1) and (state[4] == 1) and (state[8] == 1)):
            if (reward != 0):
                return None
            reward = 1
        if ((state[2] == 1) and (state[4] == 1) and (state[6] == 1)):
            if (reward != 0):
                return None
            reward = 1
        
        if ((state[0] == 2) and (state[1] == 2) and (state[2] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[3] == 2) and (state[4] == 2) and (state[5] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[6] == 2) and (state[7] == 2) and (state[8] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[0] == 2) and (state[3] == 2) and (state[6] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[1] == 2) and (state[4] == 2) and (state[7] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[2] == 2) and (state[5] == 2) and (state[8] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[0] == 2) and (state[4] == 2) and (state[8] == 2)):
            if (reward != 0):
                return None
            reward = -1
        if ((state[2] == 2) and (state[4] == 2) and (state[6] == 2)):
            if (reward != 0):
                return None
            reward = -1
        
        if (reward == 0):
            # reward = -.04
            reward = 0

        return reward
    
    def is_valid_state(self, state):
        if (self.evaluate_reward(state) is None):
            return False
        if (len((np.argwhere(state == 1))) == len((np.argwhere(state == 2)))):   
            return True 
        elif (len((np.argwhere(state == 1))) == (1 + len((np.argwhere(state == 2))))):
            return True 
        
        return False
    
    def get_state_value(self, state):
        return self.V[int("".join([str(int(i)) for i in state]), 3)]

    def evalute_value(self, state):
        original_state = np.array(state, copy=True)
        reward = self.evaluate_reward(original_state)
        empty_indices = (np.argwhere(original_state == 0))

        if ((reward == 1) or (reward == -1)):
            self.new_V[int("".join([str(int(i)) for i in original_state]), 3)] = reward
            return
        
        # If it is O's turn
        if (len((np.argwhere(original_state == 1))) == (1 + len((np.argwhere(original_state == 2))))):
            if (len(empty_indices) > 0):
                min_q = None
 
                for move_idx_o in empty_indices:
                    state[move_idx_o] = 2
                    value_for_O = self.V[int("".join([str(int(i)) for i in state]), 3)]
                    
                    if (min_q is None):
                        min_q = value_for_O
                    elif (min_q > value_for_O):
                        min_q = value_for_O

                    state[move_idx_o] = 0

                if (min_q is None):
                    min_q = 0

                self.new_V[int("".join([str(int(i)) for i in original_state]), 3)] = self.evaluate_reward(original_state) + self.gamma * min_q
        
        # If it is X's turn
        else:
            max_q = None

            for move_idx_x in empty_indices:
                state[move_idx_x] = 1
                value_for_X = self.V[int("".join([str(int(i)) for i in state]), 3)][0]

                if (max_q is None):
                    max_q = value_for_X
                elif (value_for_X > max_q):
                    max_q = value_for_X

                state[move_idx_x] = 0

            if (max_q is None):
                max_q = 0

            self.new_V[int("".join([str(int(i)) for i in state]), 3)] = reward + self.gamma * max_q
        
    def update_V(self, iterations=1):
        for l in range(iterations):
            print("Iteration " + str(l+1) + " of " + str(iterations))
            for k in range(self.possible_states):
                if (self.is_valid_state(self.states_matrix[:, k])):
                    self.evalute_value(self.states_matrix[:, k])

            self.V = np.array(self.new_V, copy=True)

    def choose_move(self, state):
        empty_indices = (np.argwhere(state == 0))
        max_q = None
        max_q_state = None

        for move_idx in empty_indices:
            state[move_idx] = 1
            value_at_move_state = self.V[int("".join([str(int(i)) for i in state]), 3)]

            if (max_q is None):
                max_q = value_at_move_state
                max_q_state = np.array(state, copy=True)
            elif (value_at_move_state > max_q):
                max_q = value_at_move_state
                max_q_state = np.array(state, copy=True)
            
            state[move_idx] = 0

        return max_q_state

    def get_valid_states(self):
        valid_array = []
        for e in range(self.possible_states):
            if (self.is_valid_state(self.states_matrix[:,e])):
                valid_array.append(e)
        return np.array(valid_array)
    
    def play_game(self):
        print("- You will play as the O's")
        print("- Follow the numbers to play")
        print("- X will start")

        print()
        print("   " + "0" + " | " + "1" + " | " + "2")
        print("----------------")
        print("   " + "3" + " | " + "4" + " | " + "5")
        print("----------------")
        print("   " + "6" + " | " + "7" + " | " + "8")
        print()  

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        state = self.choose_move(state)
        self.print_board(state)
        # print("Value at state: " + str(self.V[int("".join([str(int(i)) for i in state]), 3)]))

        while ((self.evaluate_reward(state) != 1) and (self.evaluate_reward(state) != -1) and (len((np.argwhere(state == 0))) != 0)):
            user_input = input("\nChoose your next move: ")
            empty_indices = (np.argwhere(state == 0))
            
            while (int(user_input) not in empty_indices):
                print("Choose a correct index!!!!")
                user_input = input("\nChoose your next move: ")

            state[int(user_input)] = 2
            self.print_board(state)
            # print("Value at state: " + str(self.V[int("".join([str(int(i)) for i in state]), 3)]))

            if ((self.evaluate_reward(state) == 1) or (self.evaluate_reward(state) == -1) or (len((np.argwhere(state == 0))) == 0)):
                break
            
            print("\n\nCPU next move...\n\n")
            time.sleep(1)
            state = self.choose_move(state)
            self.print_board(state)
            # print("Value at state: " + str(self.V[int("".join([str(int(i)) for i in state]), 3)]))

        reward = self.evaluate_reward(state)

        if (reward == 1):
            print("\n\nCPU WINS!!!\n\n")
        elif (reward == -1):
            print("\n\nPLAYER WINS!!!\n\n")
        else:
            print("\n\nTIE GAME!!!\n\n")

    def save_value_function(self):
        np.savetxt('Value_Function.txt', self.V, fmt='%f')

    def print_state_and_value(self, state):
        self.print_board(state)
        print("\nVALUE = " + str(self.V[int("".join([str(int(i)) for i in state]), 3)]))
        print()


if __name__ == "__main__":
    try:
        V = np.loadtxt('./Value_Function.txt', dtype=float).reshape(19683,1)
        learner = Value_Iteration_TicTacToe(start_V=V)
    except:
        learner = Value_Iteration_TicTacToe()

    # valid_states = learner.get_valid_states()
    # learner.print_state_and_value(learner.states_matrix[:, random.choice(valid_states)])
    
    # learner.update_V(iterations=200) # Train this puppy
    # learner.save_value_function()
    
    learner.play_game()