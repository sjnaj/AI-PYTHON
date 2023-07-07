init_state = (3, 3, 1)  # 传教士，野人
target_state = (0, 0, 0)


def move(states_list):
    states = []
    for states in states_list:
        state=states[len(state)-1]
        if state[0] == 1:
            if state[0] >= state[1]:
                if state[0] > 1:
                    states.append((state[0]-2, state[1], 0))
                    states.append(([state[0], state[1]-2, 0]))
                states.append(([state[0]-1, state[1], 0]))
                states.append(([state[0], state[1]-1, 0]))
        else:
            if state[0] >= state[1]:
                if 3-state[0] > 1:
                    states.append(([state[0]+2], state[1], 1))
                    states.append(([state[0], state[1]+2, 1]))
                states.append(([state[0]+1, state[1], 1]))
                states.append(([state[0], state[1]+1, 1]))
    return states

def check(states_list):
    states_list=(states for states in states_list and sta)
    

states=move([init_state])
move(states)
print(states)
