#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
    Label(  'double_white'         ,  1 ,        1 , 'lane'              , 1       , False        , False        , (240,248,255)),
    Label(  'dashed_white'         ,  2 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
    Label(  'single_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (138, 43,226)),
    Label(  'double_yellow'        ,  4 ,        4 , 'lane'              , 1       , False        , False        , (255,255,  0)),
    Label(  'dashed_yellow'        ,  5 ,        5 , 'lane'              , 1       , False        , False        , (173,255, 47)),
    Label(  'single_yellow'        ,  6 ,        6 , 'lane'              , 1       , False        , False        , (154,205, 50)),
    Label(  'double_dashed_white'  ,  7 ,        7 , 'lane'              , 1       , False        , False        , ( 30,144,255)),
    Label(  'double_dashed_yellow' ,  8 ,        8 , 'lane'              , 1       , False        , False        , (165, 42, 42)),
]

# combine dashed yellow and dashed white
labels_reduce = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
    Label(  'double_white'         ,  1 ,        1 , 'lane'              , 1       , False        , False        , (240,248,255)),
    Label(  'dashed_white'         ,  2 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
    Label(  'single_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (138, 43,226)),
    Label(  'double_yellow'        ,  4 ,        4 , 'lane'              , 1       , False        , False        , (255,255,  0)),
    Label(  'dashed_yellow'        ,  2 ,        2 , 'lane'              , 1       , False        , False        , (173,255, 47)),
    Label(  'single_yellow'        ,  5 ,        5 , 'lane'              , 1       , False        , False        , (154,205, 50)),
    Label(  'double_dashed_white'  ,  255 ,        255 , 'lane'              , 1       , False        , False        , ( 30,144,255)),
    Label(  'double_dashed_yellow' ,  255 ,        255 , 'lane'              , 1       , False        , False        , (165, 42, 42)),
]

# combine all double to class 1
# 0: background, 1:double (22000) 2: dashed(69100) 3: single_white(63500)   4: single yellow(10778)
labels_reduce_2 = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
    Label(  'double_white'         ,  1 ,        1 , 'lane'              , 1       , False        , False        , (240,248,255)),
    Label(  'dashed_white'         ,  2 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
    Label(  'single_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (138, 43,226)),
    Label(  'double_yellow'        ,  1 ,        1 , 'lane'              , 1       , False        , False        , (255,255,  0)),
    Label(  'dashed_yellow'        ,  2 ,        2 , 'lane'              , 1       , False        , False        , (173,255, 47)),
    Label(  'single_yellow'        ,  4 ,        4 , 'lane'              , 1       , False        , False        , (154,205, 50)),
    Label(  'double_dashed_white'  ,  1 ,        1 , 'lane'              , 1       , False        , False        , ( 30,144,255)),
    Label(  'double_dashed_yellow' ,  1 ,        1 , 'lane'              , 1       , False        , False        , (165, 42, 42)),
]

# combine all double to class 1, all dashed to class2, all single to class3
# 0: background, 1:double (22000) 2: dashed(69100) 3: single(74000)
labels_reduce_3 = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
    Label(  'double_white'         ,  1 ,        1 , 'lane'              , 1       , False        , False        , (240,248,255)),
    Label(  'dashed_white'         ,  2 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
    Label(  'single_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (138, 43,226)),
    Label(  'double_yellow'        ,  4 ,        1 , 'lane'              , 1       , False        , False        , (255,255,  0)),
    Label(  'dashed_yellow'        ,  5 ,        2 , 'lane'              , 1       , False        , False        , (173,255, 47)),
    Label(  'single_yellow'        ,  6 ,        3 , 'lane'              , 1       , False        , False        , (154,205, 50)),
    Label(  'double_dashed_white'  ,  7 ,        1 , 'lane'              , 1       , False        , False        , ( 30,144,255)),
    Label(  'double_dashed_yellow' ,  8 ,        1 , 'lane'              , 1       , False        , False        , (165, 42, 42)),
]

# labels_reduce_3 = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
#     Label(  'double_white'         ,  1 ,        1 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'dashed_white'         ,  2 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'single_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'double_yellow'        ,  4 ,        1 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'dashed_yellow'        ,  5 ,        2 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'single_yellow'        ,  6 ,        3 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'double_dashed_white'  ,  7 ,        1 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
#     Label(  'double_dashed_yellow' ,  8 ,        1 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
# ]


labels_lane_line = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'lane'              , 1       , False        , False        , (0  ,  0,  0)),
    Label(  'main_lane'            ,  1 ,        1 , 'lane'              , 1       , False        , False        , (255,  0,  0)),
    Label(  'alter_lane'           ,  2 ,        2 , 'lane'              , 1       , False        , False        , (255,255,255)),
    Label(  'double_white'         ,  3 ,        3 , 'lane'              , 1       , False        , False        , (240,248,255)),
    Label(  'dashed_white'         ,  4 ,        4 , 'lane'              , 1       , False        , False        , (  0,  0,255)),
    Label(  'single_white'         ,  5 ,        5 , 'lane'              , 1       , False        , False        , (138, 43,226)),
    Label(  'double_yellow'        ,  6 ,        3 , 'lane'              , 1       , False        , False        , (255,255,  0)),
    Label(  'dashed_yellow'        ,  7 ,        4 , 'lane'              , 1       , False        , False        , (173,255, 47)),
    Label(  'single_yellow'        ,  8 ,        5 , 'lane'              , 1       , False        , False        , (154,205, 50)),
    Label(  'double_dashed_white'  ,  9 ,        3 , 'lane'              , 1       , False        , False        , ( 30,144,255)),
    Label(  'double_dashed_yellow' ,  10 ,       3 , 'lane'              , 1       , False        , False        , (165, 42, 42)),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))