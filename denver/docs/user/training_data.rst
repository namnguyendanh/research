==========================
Training Data Format
==========================

We standardize the input data as a ``.csv``-format file, called ``Denver-format``, included of 3 \
columns, such as: 

+-----------+------------+-----------+
| text      |  intent    |    tags   |
+-----------+------------+-----------+


.. admonition:: **NOTE**

    .. parsed-literal::
        With the label of the NER task, ie column ``tags`` in the table above. The number of tags 
        in the column ``tags`` must be equal to the number of words of the corresponding sentence 
        in the column ``text``.

        For example:

        +-----------+------------+-------------------+
        | text      |  intent    |    tags           |
        +-----------+------------+-------------------+
        | xe con k  | query_kb   | B-object_type O O |
        +-----------+------------+-------------------+

        The number of words in sentence: ``xe con k`` is 3, so the corresponding number of tags 
        must be 3, ``B-object_type O O``.