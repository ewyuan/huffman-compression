"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for b in text:
        if b in freq_dict:
            freq_dict[b] += 1
        else:
            freq_dict[b] = 1
    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    freq_d = dict(freq_dict)
    if len(freq_d) == 1:
        return HuffmanNode(None, HuffmanNode(None),
                           HuffmanNode(freq_dict.popitem()[0]))
    else:
        queue = HuffQueue(freq_d)
        tree = queue.create_tree()
        return tree


# Helper class for huffman_tree() to build the tree from leaves to root.
class HuffQueue:
    """ A queue that orders nodes to build a Huffman tree. """

    def __init__(self, freq_dict):
        """ Initialize a new NodeQueue with the nodes from the dictionary keys
        and a paired list of weights of each node.

        @param NodeQueue self: this NodeQueue
        @param dict freq_dict: dictionary with symbol keys and weight values
        @rtype: NoneType
        """
        self._queue = []
        self._weight = []
        while freq_dict:
            pair = freq_dict.popitem()
            self._queue.append(HuffmanNode(pair[0]))
            self._weight.append(pair[1])

    def create_tree(self):
        """ Create a huffman_tree structure from the NodeQueue.

        @param NodeQueue self: this NodeQueue
        @rtype: HuffmanNode

        >>> freq = {2: 6, 3: 4}
        >>> queue = HuffQueue(freq)
        >>> tree = queue.create_tree()
        >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
        >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
        >>> tree == result1 or tree == result2
        True
        """
        while len(self._queue) != 1:
            weight = 0
            i = self._weight.index(min(self._weight))
            node_a = self._queue.pop(i)
            weight += self._weight.pop(i)
            j = self._weight.index(min(self._weight))
            node_b = self._queue.pop(j)
            weight += self._weight.pop(j)
            self._queue.append(HuffmanNode(None, node_b, node_a))
            self._weight.append(weight)
        return self._queue[0]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    symb_dict = {}
    list_of_codes = encode(tree)
    for i in range(0, len(list_of_codes), 2):
        symb_dict[list_of_codes[i]] = list_of_codes[i + 1]
    return symb_dict


# Helper function for get_codes() to add a parameter for the code itself.
def encode(tree, code=''):
    """
    Return a list with the external nodes symbol and the code
    created by the path to the node one after the other.

    @param HuffmanNode tree: root of a HuffmanNode tree
    @param str code: code linked to symbol
    @rtype: list[int, str]

    >>> t2 = HuffmanNode(3)
    >>> t1 = HuffmanNode(None, t2, HuffmanNode(None, HuffmanNode(2)))
    >>> code = encode(t1)
    >>> code
    [3, '0', 2, '10']
    """
    if tree.left is None and tree.right is None:
        return [tree.symbol, code]
    else:
        childs = [tree.left, tree.right]
        return sum([encode(childs[i], code + str(i)) for i in
                    range(len(childs)) if childs[i] is not None], [])


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    internal_list = []
    postorder_internal(tree, internal_list)
    for i in range(len(internal_list)):
        internal_list[i].number = i


# Helper function for number_nodes() to get all
# the internal nodes in the correct order.
def postorder_internal(tree, internal_list):
    """ Add to a list the internal nodes in tree
    according to postorder traversal.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @param list internal_list: list of nodes
    @rtype: None
    """
    if tree.left is not None:
        postorder_internal(tree.left, internal_list)
    if tree.right is not None:
        postorder_internal(tree.right, internal_list)
    if tree.left is not None and tree.right is not None:
        internal_list.append(tree)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    freq_d = freq_dict
    node_to_code = get_codes(tree)
    total = 0.0
    freq = 0
    for s in node_to_code:
        freq += freq_d[s]
        total += len(node_to_code[s]) * freq_d[s]
    return total / freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    code = ''
    for item in text:
        code += codes[item]
    if len(code) % 8 == 0:
        length = (len(code) // 8)
        list_of_bytes = []
        for i in range(length):
            j = (i * 8) + 8
            list_of_bytes.append(bits_to_byte(code[(i * 8):j]))
    else:
        length = (len(code) // 8) + 1
        list_of_bytes = []
        for i in range(length):
            j = (i * 8) + 8
            list_of_bytes.append(bits_to_byte(code[(i * 8):j]))
    return bytes(list_of_bytes)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    internal_list = []
    result = []
    postorder_internal(tree, internal_list)
    for tree in internal_list:
        if tree.left.left is None and tree.left.right is None:
            result.append(0)
            result.append(tree.left.symbol)
        else:
            result.append(1)
            result.append(tree.left.number)
        if tree.right.left is None and tree.right.right is None:
            result.append(0)
            result.append(tree.right.symbol)
        else:
            result.append(1)
            result.append(tree.right.number)
    return bytes(result)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> tree_example = HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, \
    None, None), HuffmanNode(12, None, None)), HuffmanNode(None, \
    HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    >>> tree_example == generate_tree_general(lst, 2)
    True
    """
    if node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 0:
        node = HuffmanNode(None, HuffmanNode(node_lst[root_index].l_data),
                           HuffmanNode(node_lst[root_index].r_data))
    elif node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 1:
        node = HuffmanNode(None, HuffmanNode(node_lst[root_index].l_data),
                           generate_tree_general(node_lst,
                                                 node_lst[root_index].r_data))
    elif node_lst[root_index].l_type == 1 and node_lst[root_index].r_type == 0:
        node = HuffmanNode(None,
                           generate_tree_general(node_lst,
                                                 node_lst[root_index].l_data),
                           HuffmanNode(node_lst[root_index].r_data))
    else:
        node = HuffmanNode(None,
                           generate_tree_general(node_lst,
                                                 node_lst[root_index].l_data),
                           generate_tree_general(node_lst,
                                                 node_lst[root_index].r_data))
    node.number = root_index
    return node


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 20, 0, 16), ReadNode(0, 26, 0, 23), \
    ReadNode(0, 15, 1, 0), ReadNode(1, 0, 1, 0)]
    >>> tree_example = HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(20, None, None), HuffmanNode(16, None, None)), \
    HuffmanNode(None, HuffmanNode(15, None, None), HuffmanNode(None, \
    HuffmanNode(26, None, None), HuffmanNode(23, None, None))))
    >>> tree_example == generate_tree_postorder(lst, 3)
    True
    """
    if node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 0:
        node = HuffmanNode(None, HuffmanNode(node_lst[root_index].l_data),
                           HuffmanNode(node_lst[root_index].r_data))
    elif node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 1:
        node = HuffmanNode(None, HuffmanNode(node_lst[root_index].l_data),
                           generate_tree_postorder(node_lst, (root_index - 1)))
    elif node_lst[root_index].l_type == 1 and node_lst[root_index].r_type == 0:
        node = HuffmanNode(None,
                           generate_tree_postorder(node_lst, (root_index - 1)),
                           HuffmanNode(node_lst[root_index].r_data))
    else:
        right = generate_tree_postorder(node_lst, (root_index - 1))
        number_nodes(right)
        new_index = (root_index - 1) - (right.number + 1)
        node = HuffmanNode(None,
                           generate_tree_postorder(node_lst, new_index), right)
    number_nodes(node)
    return node


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    code_str = ''
    for byte in text:
        code_str += byte_to_bits(byte)
    code_str = code_str
    current = tree
    byte_lst = []
    for i in range(len(code_str)):
        current = navigate_tree(current, code_str[i])
        if current.symbol is not None:
            byte_lst.append(current.symbol)
            current = tree
    return bytes(byte_lst[:size])


def navigate_tree(tree, branch):
    """ Return a tree based on its branch. '0' represents left tree,
    '1' represents right tree

    @param HuffmanNode tree: a HuffmanNode tree
    @param str branch: a '0' or '1'
    @rtype: HuffmanNode

    >>> tree = HuffmanNode(None, HuffmanNode(5), HuffmanNode(6))
    >>> navigate_tree(tree, '0')
    HuffmanNode(5, None, None)
    """
    if branch == '0':
        return tree.left
    elif branch == '1':
        return tree.right


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        # tree = generate_tree_postorder(node_lst, root_index)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions


def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> left = HuffmanNode(None, HuffmanNode(None, HuffmanNode(10), HuffmanNode(5)), HuffmanNode(25))
    >>> right = HuffmanNode(None, HuffmanNode(26), HuffmanNode(None, HuffmanNode(3), HuffmanNode(6)))
    >>> tree =  HuffmanNode(None, left, right)
    >>> freq = {10: 10, 5: 90, 25: 500, 26: 2, 3: 20, 6: 30}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    """
    freq_dict_copy = dict(freq_dict)
    code_list = sort_codes(get_codes(tree))
    i = 0
    while i < len(code_list):
        current = tree
        for bit in code_list[i]:
            current = navigate_tree(current, bit)
        highest_freq_symbol = max(freq_dict_copy, key=freq_dict_copy.get)
        current.symbol = highest_freq_symbol
        del freq_dict_copy[highest_freq_symbol]
        i += 1


def sort_codes(code_dict):
    """ Return a list of codes based on it's int value and length.

    @param dict(int,str) code_dict: a dictionary with codes
    @rtype: list(str)

    >>> code_dict = {1: "11010", 3: "11011", 4: "1100", 10: "111", 12: "00"}
    >>> sort_codes(code_dict)
    ['00', '111', '1100', '11010', '11011']
    """
    list_ = []
    min_length = len(min(code_dict.values(), key=len))
    max_length = len(max(code_dict.values(), key=len))
    i = min_length
    while i != max_length + 1:
        list_with_same_len = []
        for key in code_dict:
            if len(code_dict[key]) == i:
                list_with_same_len.append(code_dict[key])
        i += 1
        list_with_same_len = sorted(list_with_same_len)
        list_ += list_with_same_len
    return list_


if __name__ == "__main__":
    # import python_ta
    # python_ta.check_all(config="huffman_pyta.txt")
    # import doctest
    # doctest.testmod()
    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
