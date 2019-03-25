# data taken from: https://www.kaggle.com/uciml/mushroom-classification, then converted into dictionaries

schema = dict(

    cap_shape = dict(bell='b', conical='c', convex='x', flat='f',  knobbed='k', sunken='s'),

    cap_surface = dict(fibrous='f', grooves='g', scaly='y', smooth='s'),

    cap_color = dict(brown='n', buff='b', cinnamon='c', gray='g', green='r', pink='p', purple='u', red='e', white='w', yellow='y'),

    bruises = dict(bruises='t', no='f'),

    odor = dict(almond='a', anise='l', creosote='c', fishy='y', foul='f', musty='m', none='n', pungent='p', spicy='s'),

    gill_attachment = dict(attached='a', descending='d', free='f', notched='n'),

    gill_spacing = dict(close='c', crowded='w', distant='d'),

    gill_size = dict(broad='b', narrow='n'),

    gill_color = dict(black='k', brown='n', buff='b', chocolate='h', gray='g',  green='r', orange='o', pink='p', purple='u', red='e', white='w', yellow='y'),

    stalk_shape = dict(enlarging='e', tapering='t'),

    stalk_root = dict(bulbous='b', club='c', cup='u', equal='e', rhizomorphs='z', rooted='r', missing="?"),

    stalk_surface_above_ring = dict(fibrous='f', scaly='y', silky='k', smooth='s'),

    stalk_surface_below_ring = dict(fibrous='f', scaly='y', silky='k', smooth='s'),

    stalk_color_above_ring = dict(brown='n', buff='b', cinnamon='c', gray='g', orange='o', pink='p', red='e', white='w', yellow='y'),

    stalk_color_below_ring = dict(brown='n', buff='b', cinnamon='c', gray='g', orange='o', pink='p', red='e', white='w', yellow='y'),

    veil_type = dict(partial='p', universal='u'),

    veil_color = dict(brown='n', orange='o', white='w', yellow='y'),

    ring_number = dict(none='n', one='o', two='t'),

    ring_type = dict(cobwebby='c', evanescent='e', flaring='f', large='l', none='n', pendant='p', sheathing='s', zone='z'),

    spore_print_color = dict(black='k', brown='n', buff='b', chocolate='h', green='r', orange='o', purple='u', white='w', yellow='y'),

    population = dict(abundant='a', clustered='c', numerous='n', scattered='s', several='v', solitary='y'),

    habitat = dict(grasses='g', leaves='l', meadows='m', paths='p', urban='u', waste='w', woods='d'),

)
