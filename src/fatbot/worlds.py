from math import pi


# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
""" A collection of fatbot worlds (args) """
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -

default = dict(
    name =              'default_world',
    horizon =           0,      # anything less than or equal to 0 is same as inf
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)
    world_info = dict(
        X_RANGE =       1,  # [+/-] world x dimension
        Y_RANGE =       1,  # [+/-] world y dimension
        N_BOTS  =       1,  # no of fat-bots
        BOT_RADIUS =    1,  # fat-bot body radius
        SCAN_RADIUS =   1,  # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        TARGET_RADIUS = 0.0,
        SAFE_DISTANCE = 1,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        SPEED_LIMIT =   1,  # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,  # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 1/pi  # choose based on scan distance, use form: n/pi 
        ), 
    bot_info = dict(
        NAME =     ['default_bot'], 
        COLOR =    ['black'], 
        MARKER  =  ['.',   ], 
        ), 

    # Initial state Distribution:
    #   a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots 
    initial_states = [ [  (0,0),  ], ],
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove





world_dual_test = dict(

    name =              'world_dual_test',
    
    horizon =           200, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       10,       # [+/-] world x dimension
        Y_RANGE =       10,       # [+/-] world y dimension
        N_BOTS  =       2,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   8,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 0.5,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 20/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['brown',     'blue'], 
        COLOR =    ['tab:brown', 'tab:blue'], 
        MARKER  =  ['.',         '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-6,-6),  (6, 6),   ],
        [ (6, 6),   ( -6, -6),   ],
        [ (6, -6),   ( -6, 6),   ],
        [ (-6, 6),   ( 6, -6),   ],
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove

world_triple_test = dict(

    name =              'world_triple_test',
    
    horizon =           300, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       10,       # [+/-] world x dimension
        Y_RANGE =       10,       # [+/-] world y dimension
        N_BOTS  =       3,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   8,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 0.5,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 20/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['brown',     'blue',        'purple'    ], 
        COLOR =    ['tab:brown', 'tab:blue',    'tab:purple'], 
        MARKER  =  ['.',         '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-6,-6),  (6, 6),     (6,0) ],
        [ (6, 6),   ( -6, -6),  (-6,0) ],
        [ (6, -6),   ( -6, 6),   (0,-6)],
        [ (-6, 6),   ( 6, -6),   (0,6)],
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove

world_quad_test = dict(

    name =              'world_quad_test',
    
    horizon =           400, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       10,       # [+/-] world x dimension
        Y_RANGE =       10,       # [+/-] world y dimension
        N_BOTS  =       4,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   8,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 0.5,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 20/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['brown',     'blue',        'purple',        'green'        ], 
        COLOR =    ['tab:brown', 'tab:blue',    'tab:purple',    'tab:green'], 
        MARKER  =  ['.',         '.', '.',         '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-6,-6),  (6, 6),      (6,0), (-6,0) ],
        [ (6, 6),   ( -6, -6),  (-6,0), (6,0) ],
        [ (6, -6),   ( -6, 6),   (0,-6), (0, 6)],
        [ (-6, 6),   ( 6, -6),   (0,6),  (0, -6)],
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove





world_x4 = dict(

    name =              'world_x4',
    
    horizon =           500, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       20,       # [+/-] world x dimension
        Y_RANGE =       20,       # [+/-] world y dimension
        N_BOTS  =       4,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   15,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        TARGET_RADIUS = 0.0,
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',      'G',      'Y'], 
        COLOR =    ['red',   'blue',     'green',  'gold'], 
        MARKER  =  ['.',         '.',       '.',        '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),   ],
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove

world_x5 = dict(

    name =              'world_x5',
    
    horizon =           500, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       20,       # [+/-] world x dimension
        Y_RANGE =       20,       # [+/-] world y dimension
        N_BOTS  =       5,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   15,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (0,0) ],
        
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove

world_x6 = dict(

    name =              'world_x6',
    
    horizon =           800, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       20,       # [+/-] world x dimension
        Y_RANGE =       20,       # [+/-] world y dimension
        N_BOTS  =       6,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   15,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        TARGET_RADIUS = 0.0,
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P',       'C'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple',  'cyan'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (18,0),  (-18,0) ],
        
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove


world_x7 = dict(

    name =              'world_x7',
    
    horizon =           800, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       20,       # [+/-] world x dimension
        Y_RANGE =       20,       # [+/-] world y dimension
        N_BOTS  =       7,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   15,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P',       'C',      'M'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple',  'cyan', 'magenta'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.',       '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (18,0),  (-18,0), (0,18) ],
        
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove

world_x8 = dict(

    name =              'world_x8',
    
    horizon =           1000, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    False, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       20,       # [+/-] world x dimension
        Y_RANGE =       20,       # [+/-] world y dimension
        N_BOTS  =       8,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   15,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 0.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P',       'C',      'M',      'K'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple',  'cyan', 'magenta',  'pink'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.', '.', '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (18,0),  (-18,0), (0,18), (0, -18) ],
        
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove












world_o4 = dict(

    name =              'world_o4',
    
    horizon =           500, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    True, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       25,       # [+/-] world x dimension
        Y_RANGE =       25,       # [+/-] world y dimension
        N_BOTS  =       4,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   20,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        TARGET_RADIUS = 2.0,
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',      'G',      'Y'], 
        COLOR =    ['red',   'blue',     'green',  'gold'], 
        MARKER  =  ['.',         '.',       '.',        '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),   ],
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove



world_o5 = dict(

    name =              'world_o5',
    
    horizon =           500, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    True, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       25,       # [+/-] world x dimension
        Y_RANGE =       25,       # [+/-] world y dimension
        N_BOTS  =       5,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   20,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        TARGET_RADIUS = 4.0,
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (0,0) ],
        
        ],
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove


world_o6 = dict(

    name =              'world_o6',
    
    horizon =           800, # anything less than or equal to 0 is same as inf
    
    enable_imaging =    True, # if true, generates sensor images (o-ray, x-ray, d-ray)

    world_info = dict(
        X_RANGE =       25,       # [+/-] world x dimension
        Y_RANGE =       25,       # [+/-] world y dimension
        N_BOTS  =       6,       # no of fat-bots
        BOT_RADIUS =    1,     # fat-bot body radius
        SCAN_RADIUS =   20,   # scannig radius of onboard sensor (neighbourhood) - considers the center point of neighbours as within or outside range
        SAFE_DISTANCE = 1.0,  # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
        TARGET_RADIUS = 5.0,
        SPEED_LIMIT =   1,    # [+/-] upper speed (roll, pitch) limit of all robots
        DELTA_SPEED =   0,    # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
        SENSOR_RESOULTION = 40/pi  # choose based on scan distance, use form: n/pi 
        ), 

    bot_info = dict(
        NAME =     ['R',     'B',        'G',      'Y',      'P',       'C'], 
        COLOR =    ['red',   'blue',     'green',  'gold',   'purple',  'cyan'], 
        MARKER  =  ['.',         '.',       '.',        '.', '.', '.'], 
        ), 

    initial_states = [ # a list of list of 2-tuples, each item in this list is a state which is just (x,y) of all bots
            # brown     # blue   
        [ (-18,-18),  (18, 18), (18, -18), (-18, 18),  (18,0),  (-18,0) ],
        
        ],
    
    seed = None, # for choosing initial states uniformly 
    ) # keep default - do not remove
