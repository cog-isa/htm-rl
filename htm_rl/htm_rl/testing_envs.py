from htm_rl.envs.mdp import GridworldMdpGenerator, Mdp


def make_mdp_passage(cell_gonality, path, seed, clockwise_action: bool = False):
    initial_state = (0, 0)
    current_cell = 0
    cell_transitions = []
    for direction in path:
        cell_transitions.append((current_cell, direction, current_cell + 1))
        current_cell += 1

    generator = GridworldMdpGenerator(cell_gonality)
    return generator.generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way(start_direction, seed, clockwise_action: bool = False):
    """
    ######
    #01###
    #234##
    ##567#
    ###89#
    ######
    """
    initial_state = (0, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 2),  # c0 . c2
        (1, 3, 3),  # c1 . c3
        (2, 0, 3),  # c2 > c3
        (3, 0, 4),  # c3 > c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 6),  # c4 > c6
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
        (6, 3, 8),  # c6 . c8
        (7, 3, 9),  # c7 . c9
        (8, 0, 9),  # c8 > c9
    ]

    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way_v0(start_direction, seed, clockwise_action: bool = False):
    """
    #####
    #012#
    #3#4#
    #567#
    #####
    """
    initial_state = (0, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 3),  # c0 . c3
        (1, 0, 2),  # c1 > c2
        (2, 3, 4),  # c2 . c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 7),  # c4 . c7
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
    ]
    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way_v1(start_direction, seed, clockwise_action: bool = False):
    """
    #######
    #012###
    #3#4###
    #56789#
    ###0#1#
    ###234#
    #######
    """
    initial_state = (0, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 3),  # c0 . c3
        (1, 0, 2),  # c1 > c2
        (2, 3, 4),  # c2 . c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 7),  # c4 . c7
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
        (7, 0, 8),  # c7 > c8
        (7, 3, 10),  # c7 . c10
        (8, 0, 9),  # c8 > c9
        (9, 3, 11),  # c9 . c11
        (10, 3, 12),  # c10 . c12
        (11, 3, 14),  # c11 . c14
        (12, 0, 13),  # c12 > c13
        (13, 0, 14),  # c13 > c14
    ]
    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)


def make_mdp_multi_way_v2(start_direction, seed, clockwise_action: bool = False):
    """
    #########
    #012#####
    #3#4#####
    #567890##
    ###12####
    ###3#890#
    ###467###
    ###5#####
    #########
    """
    initial_state = (7, start_direction)
    cell_transitions = [
        (0, 0, 1),  # c0 > c1
        (0, 3, 3),  # c0 . c3
        (1, 0, 2),  # c1 > c2
        (2, 3, 4),  # c2 . c4
        (3, 3, 5),  # c3 . c5
        (4, 3, 7),  # c4 . c7
        (5, 0, 6),  # c5 > c6
        (6, 0, 7),  # c6 > c7
        (7, 0, 8),  # c7 > c8
        (7, 3, 11),  # c7 . c11
        (8, 0, 9),  # c8 > c9
        (8, 3, 12),  # c8 > c12
        (9, 0, 10),  # c9 . c10
        (11, 0, 12),  # c11 . c12
        (11, 3, 13),  # c11 . c13
        (13, 3, 14),  # c13 . c14
        (14, 0, 16),  # c14 . c16
        (14, 3, 15),  # c14 . c15
        (16, 0, 17),  # c16 > c17
        (17, 1, 18),  # c17 > c18
        (18, 0, 19),  # c18 > c19
        (19, 0, 20),  # c19 > c20
    ]
    return GridworldMdpGenerator(4).generate_env(Mdp, initial_state, cell_transitions, clockwise_action, seed)
