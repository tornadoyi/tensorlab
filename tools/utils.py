import re
import os
import shutil


def search_keyword(text, *args):
    ex = ""

    for i in xrange(len(args)):
        s = args[i]
        ex += s
        if i < len(args) - 1:
            ex += "\s+"

    it = re.search(ex, text)
    return it.span() if it != None else it



def find_pair(content, start, pos, neg):
    assert content[start] == pos

    count = 0
    find_index = None
    for i in xrange(start, len(content)):
        s = content[i]
        if s == pos: count += 1
        elif s == neg: count -= 1
        if count == 0:
            find_index = i
            break

    return find_index

def remove_path_files(path):
    cur_path = os.getcwd()
    os.chdir(path)
    os.system("rm -rf *")
    os.chdir(cur_path)