#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "ririhedou@gmail.com"

import os

# Support classes
class Candidate(object):
    """
    Candidates are for the image and html source files
    """
    def __init__(self, idx, web_img, web_source, mobile_img, mobile_source):
        self.idx = idx
        self.web_img = web_img
        self.web_source = web_source
        self.mobile_img = mobile_img
        self.mobile_source = mobile_source


#the structure for crawling data
class CrawlCandidate(object):
    """
    Candidates are for the image and html source file in crawling
    """
    def __init__(self, idx, img, source, redirect):
        self.idx = idx
        self.web_img = img
        self.web_source = source
        self.redirect = redirect


class Feature(object):
    """
    Feature is used for extract feature vector
    """

    def __init__(self):
        self.nlp_text = list()
        self.img_text = list()
        self.form_text = list()
        self.form_num = 0

    def update_nlp_text(self, text):
        self.nlp_text.extend(text)

    def update_img_text(self, text):
        self.img_text.extend(text)

    def update_structure(self):
        pass


# Support functions
def read_pngs_sources_from_directory(dire):
    files = os.listdir(dire)
    if not dire.endswith('/'):
        dire = dire + '/'

    idxs = list()
    candidates = list()
    for f in files:
        if f.startswith('.'):
            continue
        idx = f.split('.')[0]
        if not idx in idxs:
            web_img = dire + idx + '.web.screen.png'
            web_source = dire + idx + '.web.source.html'
            mobile_img = dire + idx + '.mobile.screen.png'
            mobile_source = dire + idx + '.mobile.source.html'
            can = Candidate(idx, web_img, web_source, mobile_img, mobile_source)
            candidates.append(can)
            idxs.append(idx)
    return candidates


def read_pngs_sources_from_multiple_directories(dire_list):
    cans = list()
    for i in dire_list:
        can = read_pngs_sources_from_directory(i)
        cans.extend(can)
    return cans



"""
('chinese-facebook.de..screen.png', 'chinese-facebook.de..screen')
('my-facebook.org..source.txt', 'my-facebook.org..source')
('fakebook.pro..redirect', 'fakebook.pro.')

"""
def read_candidates_from_crawl_data(dire):
    if not dire.endswith('/'):
        dire += '/'

    IDX_list = set()
    for f in os.listdir(dire):

        idx = f.split('..')[0]
        IDX_list.add(idx)

    crawl_candidate_list = list()

    for i in IDX_list:
        img = dire + i + '..screen.png'
        source = dire + i + '..source.txt'
        redirect = dire + i + '..redirect'
        crawl_candidate_list.append(CrawlCandidate(i, img, source, redirect))
    return crawl_candidate_list
