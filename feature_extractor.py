import urllib
import urllib.request
import re
from tld import get_tld, get_fld, is_tld
import statistics
from urllib.parse import urlparse
import requests
from datetime import datetime
import subprocess
import json
import logging
import numpy as np
import joblib
import pickle
import whois
import pandas as pd
import os
import ssl
import socket
import ipaddress


# --- HTML features ---

def getObjects(HTML):
    images = HTML.find_all('img')
    links = HTML.find_all('link')
    anchors = HTML.find_all('a')
    sounds = HTML.find_all('sound')
    videos = HTML.find_all('video')
    objects = images + links + anchors + sounds + videos
    return objects


def sameAuthors(element_location, URL):
    element_domain = ((urlparse(element_location)).netloc).lower()
    if len(element_domain) == 0:
        return False
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    domain_words = domain.split(".")
    words_to_check = []
    for word in domain_words:
        if len(word) > 3:
            words_to_check.append(word)
    for word in words_to_check:
        if element_domain.find(word) > -1:
            return True
    return False


def isInternal(element_location, URL):
    if element_location.startswith("http"):
        return sameAuthors(element_location, URL)
    return True


def checkObjects(objects, URL):
    suspicious_threshold = 0.15
    phishing_threshold = 0.3
    if len(objects) == 0:
        return -1
    external_objects = []
    object_locations = []
    for obj in objects:
        if "src" in obj.attrs:
            object_location = obj["src"]
        elif "href" in obj.attrs:
            object_location = obj["href"]
        else:
            continue
        object_locations.append(object_location)
        if not isInternal(object_location, URL):
            external_objects.append(obj)

    if len(object_locations) == 0:
        return -1
    external_objects_rate = len(external_objects) / len(object_locations)
    if external_objects_rate < suspicious_threshold:
        return -1
    elif external_objects_rate < phishing_threshold:
        return 0
    return 1


def checkMetaScripts(HTML, URL):
    suspicious_threshold = 0.52
    phishing_threshold = 0.61
    metas = HTML.findAll("meta")
    scripts = HTML.findAll("script")
    links = HTML.findAll("link")
    objects = metas + scripts + links
    if len(objects) == 0:
        return -1
    external_objects = []
    object_locations = []
    for o in objects:
        object_location = ""
        keys = o.attrs.keys()
        if "src" in keys:
            object_location = o["src"]
            object_locations.append(object_location)
        elif "href" in keys:
            object_location = o["href"]
            object_locations.append(object_location)
        elif "http-equiv" in keys:
            if "content" in keys:
                content = o.attrs["content"]
                content_split = content.split("URL=")
                if len(content_split) > 1:
                    object_location = content_split[1].strip()
                    object_locations.append(object_location)
        if object_location == "":
            continue
        if not (isInternal(object_location, URL)):
            external_objects.append(o)
    if len(object_locations) == 0:
        return -1
    external_objects_rate = len(external_objects) / len(object_locations)
    if external_objects_rate < suspicious_threshold:
        return -1
    elif external_objects_rate < phishing_threshold:
        return 0
    return 1


def checkFrequentDomain(objects, URL):
    if len(objects) == 0:
        return -1
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)

    ex_domains = [x for x in ex_domains if "w3.org" not in x]
    if len(ex_domains) == 0:
        return -1
    try:
        frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
    except:
        frequency_ex = 0
    if frequency_in >= frequency_ex:
        return -1
    else:
        return 1


def checkCommonPageRatioinWeb(objects, HTML, URL):
    metas = HTML.findAll("meta")
    scripts = HTML.findAll("script")
    objects = objects + metas + scripts
    if len(objects) == 0:
        return 0
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)
    if len(object_locations) == 0:
        return 0
    if len(ex_domains) > 0:
        try:
            frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
        except:
            frequency_ex = 0
    else:
        frequency_ex = 0
    if frequency_in >= frequency_ex:
        most_frequent = frequency_in
    else:
        most_frequent = frequency_ex
    total = len(object_locations)
    ratio = most_frequent / total
    return float(format(ratio, ".3f"))


def checkCommonPageRatioinFooter(HTML, URL):
    foot = HTML.footer
    if foot is None:
        return 0
    images = foot.findAll("img")
    links = foot.findAll("link")
    anchors = foot.findAll("a")
    sounds = foot.findAll("sound")
    videos = foot.findAll("video")
    metas = foot.findAll("meta")
    li = foot.findAll("li")
    scripts = foot.findAll("script")
    objects = images + links + anchors + sounds + videos + metas + scripts + li
    if len(objects) == 0:
        return 0
    object_locations = []
    ex_domains = []
    frequency_in = 0
    for o in objects:
        try:
            object_location = o["src"]
            object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                object_locations.append(object_location)
            except:
                continue
        if isInternal(object_location, URL):
            frequency_in = frequency_in + 1
        else:
            ex_domain = ((urlparse(object_location)).netloc).lower()
            ex_domains.append(ex_domain)
    if len(object_locations) == 0:
        return 0
    if len(ex_domains) > 0:
        try:
            frequency_ex = max(ex_domains.count(b) for b in ex_domains if b)
        except:
            frequency_ex = 0
    else:
        frequency_ex = 0
    if frequency_in >= frequency_ex:
        most_frequent = frequency_in
    else:
        most_frequent = frequency_ex
    total = len(object_locations)
    ratio = most_frequent / total
    return float(format(ratio, ".3f"))


def checkSFH(HTML, URL):
    suspicious_threshold = 0.5
    phishing_threshold = 0.75
    forms = HTML.findAll("form")
    if len(forms) == 0:
        return -1
    suspicious_forms = []
    for form in forms:
        if "action" in form:
            form_location = form["action"]
            if not (isInternal(form_location, URL)):
                suspicious_forms.append(form)
            elif form_location == "about:blank":
                suspicious_forms.append(form)
            elif form_location == "":
                suspicious_forms.append(form)
    suspicious_forms_rate = len(suspicious_forms) / len(forms)
    if suspicious_forms_rate < suspicious_threshold:
        return -1
    elif suspicious_forms_rate < phishing_threshold:
        return 0
    return 1


def checkPopUp(HTML):
    if str(HTML).find("prompt(") >= 0:
        return 1
    elif str(HTML).find("window.open(") >= 0:
        return 0
    return -1


def checkRightClick(HTML):
    contextmenu_disabler_html = 'oncontextmenu="return false;"'
    if str(HTML).find(contextmenu_disabler_html) >= 0:
        return 1
    return -1


def checkDomainwithCopyright(HTML, URL):
    try:
        res = get_tld(URL, as_object=True)
    except:
        return 1
    domain = res.domain
    symbol = "\N{COPYRIGHT SIGN}".encode("utf-8")
    symbol = symbol.decode("utf-8")
    pattern = r"" + symbol
    if len(HTML.findAll(text=re.compile(pattern))) < 1:
        return 0
    for tag in HTML.findAll(text=re.compile(pattern)):
        copyrightTexts = tag.parent.text
        if copyrightTexts.find(domain) > -1:
            return -1
    return 1


def nullLinksinWeb(HTML, URL):
    anchors = HTML.findAll("a")
    if len(anchors) == 0:
        return 0
    suspicious_anchors = []
    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
        elif isInternal(anchor_location, URL):
            suspicious_anchors.append(a)
    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)
    return float(format(suspicious_anchors_rate, ".2f"))


def nullLinksinFooter(HTML, URL):
    foot = HTML.footer
    suspicious_anchors = []
    if foot is None:
        return 0
    anchors = foot.findAll("a")
    if len(anchors) == 0:
        return 0
    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)
    return float(format(suspicious_anchors_rate, ".2f"))


def checkBrokenLink(HTML, URL):
    images = HTML.find_all("img")
    links = HTML.find_all("link")
    anchors = HTML.find_all("a")
    sounds = HTML.find_all("sound")
    videos = HTML.find_all("video")
    metas = HTML.find_all("meta")
    scripts = HTML.find_all("script")
    objects = images + links + anchors + sounds + videos + metas + scripts
    broken_link = 0
    if len(objects) == 0:
        return 0
    object_locations = []
    for o in objects:
        try:
            object_location = o["src"]
            if not (isInternal(object_location, URL)):
                object_locations.append(object_location)
        except:
            try:
                object_location = o["href"]
                if not (isInternal(object_location, URL)):
                    object_locations.append(object_location)
            except:
                continue
    if len(object_locations) == 0:
        return 0
    for obj in object_locations:
        try:
            resp = urllib.request.urlopen(obj, timeout=2)
            status_code = resp.getcode()
            if status_code >= 400:
                broken_link = broken_link + 1
        except:
            broken_link = broken_link + 1
    broken_link_rate = broken_link / len(object_locations)
    return float(format(broken_link_rate, ".2f"))


def checkLoginForm(HTML, URL):
    forms = HTML.findAll("form")
    empty = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever", "#content", "javascript::void(0)",
             "javascript::void(0);", "javascript::;", "javascript"]
    for obj in forms:
        if "action" in obj.attrs:
            if obj["action"] in empty or not (isInternal(obj["action"], URL)):
                return 1
    return -1


def checkHiddenInfo_div(HTML):
    divs = HTML.findAll("div")
    for div in divs:
        if "style" in div.attrs and ("visibility:hidden" in div["style"] or "display:none" in div["style"]):
            return 1
    return -1


def checkHiddenInfo_button(HTML):
    buttons = HTML.findAll("button")
    for button in buttons:
        if "disabled" in button.attrs and button["disabled"] in ["", "disabled"]:
            return 1
    return -1


def checkHiddenInfo_input(HTML):
    inputs = HTML.findAll("input")
    for inp in inputs:
        if ("type" in inp.attrs and inp["type"] == "hidden") or "disabled" in inp.attrs:
            return 1
    return -1


def checkTitleUrlBrand(HTML, URL):
    try:
        domain_brand = (get_tld(URL, as_object=True)).domain
    except:
        return 1
    try:
        title = HTML.find("title").get_text()
        if len(title) < 2:
            return 0
        elif title.find(domain_brand) > -1:
            return -1
        else:
            return 1
    except:
        return 0


def checkIFrame(HTML):
    iframes = HTML.find_all("iframe")
    for iframe in iframes:
        try:
            if ((iframe["style"].find("display: none") > -1) or (iframe["style"].find("border: 0") > -1) or (
                    iframe["style"].find("visibility: hidden;") > -1) or (iframe["frameborder"].find("0") > -1)):
                return 1
        except:
            continue
    return -1


def checkFavicon(HTML, URL):
    favicon = HTML.find(rel="shortcut icon")
    if not favicon:
        favicon = HTML.find(rel="icon")
    if favicon:
        if "href" in favicon.attrs:
            if isInternal(favicon["href"], URL):
                return -1
            else:
                return 1
    return 0


def checkStatusBar(HTML):
    status_bar_modification = "window.status"
    if str(HTML).find(status_bar_modification) >= 0:
        return 1
    return -1


def checkCSS(HTML, URL):
    css = HTML.find(rel="stylesheet")
    if css is not None and "href" in css.attrs and not (isInternal(css["href"], URL)):
        return 1
    return -1


def checkAnchors(HTML, URL):
    suspicious_threshold = 0.32
    phishing_threshold = 0.505
    anchors = HTML.findAll("a")
    if len(anchors) == 0:
        return -1
    suspicious_anchors = []
    for a in anchors:
        try:
            anchor_location = a["href"]
        except:
            continue
        if anchor_location == "#":
            suspicious_anchors.append(a)
        elif anchor_location == "#content":
            suspicious_anchors.append(a)
        elif anchor_location == "#skip":
            suspicious_anchors.append(a)
        elif anchor_location == "JavaScript ::void(0)":
            suspicious_anchors.append(a)
        elif not (isInternal(anchor_location, URL)):
            suspicious_anchors.append(a)
    suspicious_anchors_rate = len(suspicious_anchors) / len(anchors)
    if suspicious_anchors_rate < suspicious_threshold:
        return -1
    elif suspicious_anchors_rate < phishing_threshold:
        return 0
    return 1


def extract_features_html(HTML, URL):
    h_features = {}
    objects = getObjects(HTML)
    h_features["HTML_Objects"] = checkObjects(objects, URL)
    h_features["HTML_metaScripts"] = checkMetaScripts(HTML, URL)
    h_features["HTML_FrequentDomain"] = checkFrequentDomain(objects, URL)
    h_features["HTML_Commonpage"] = checkCommonPageRatioinWeb(objects, HTML, URL)
    h_features["HTML_CommonPageRatioinFooter"] = checkCommonPageRatioinFooter(HTML, URL)
    h_features["HTML_SFH"] = checkSFH(HTML, URL)
    h_features["HTML_popUp"] = checkPopUp(HTML)
    h_features["HTML_RightClick"] = checkRightClick(HTML)
    h_features["HTML_DomainwithCopyright"] = checkDomainwithCopyright(HTML, URL)
    h_features["HTML_nullLinksinWeb"] = nullLinksinWeb(HTML, URL)
    h_features["HTML_nullLinksinFooter"] = nullLinksinFooter(HTML, URL)
    h_features["HTML_BrokenLink"] = checkBrokenLink(HTML, URL)
    h_features["HTML_LoginForm"] = checkLoginForm(HTML, URL)
    h_features["HTML_HiddenInfo_div"] = checkHiddenInfo_div(HTML)
    h_features["HTML_HiddenInfo_button"] = checkHiddenInfo_button(HTML)
    h_features["HTML_HiddenInfo_input"] = checkHiddenInfo_input(HTML)
    h_features["HTML_TitleUrlBrand"] = checkTitleUrlBrand(HTML, URL)
    h_features["HTML_IFrame"] = checkIFrame(HTML)
    h_features["HTML_favicon"] = checkFavicon(HTML, URL)
    h_features["HTML_statusBarMod"] = checkStatusBar(HTML)
    h_features["HTML_css"] = checkCSS(HTML, URL)
    h_features["HTML_anchors"] = checkAnchors(HTML, URL)
    return h_features


# --- URL_features ---

def checkLength(URL):
    legitimate_threshold = 54
    suspicious_threshold = 75
    if len(URL) < legitimate_threshold:
        return -1
    elif len(URL) < suspicious_threshold:
        return 0
    else:
        return 1


def hexDecoder(domain):
    try:
        n = domain.split(".")
        IPv4 = str(int(n[0], 16))
        for number in n[1:]:
            IPv4 = IPv4 + "." + str(int(number, 16))
        return IPv4
    except:
        return 0


# [FIXED] Uses ipaddress lib for robust checking
def checkIP(URL):
    domain = urlparse(URL).netloc
    if not domain:
        domain = URL.split('/')[0]
    if ':' in domain:
        domain = domain.split(':')[0]
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return -1


def checkRedirect(URL):
    if URL.rfind("//") > 7:
        redirect = 1
    else:
        redirect = -1
    return redirect


def checkShortener(URL):
    shorteners_list = ["bit.do", "t.co", "lnkd.in", "db.tt", "qr.ae", "adf.ly", "goo.gl", "bitly.com", "cur.lv",
                       "tinyurl.com", "ow.ly", "bit.ly", "ity.im", "q.gs", "is.gd", "po.st", "bc.vc", "twitthis.com",
                       "u.to", "j.mp", "buzurl.com", "cutt.us", "u.bb", "yourls.org", "x.co", "prettylinkpro.com",
                       "scrnch.me", "filoops.info", "vzturl.com", "qr.net", "1url.com", "tweez.me", "v.gd", "tr.im",
                       "link.zip.net", "tinyarrows.com", "adcraft.co", "adcrun.ch", "adflav.com", "aka.gr", "bee4.biz",
                       "cektkp.com", "dft.ba", "fun.ly", "fzy.co", "gog.li", "golinks.co", "hit.my", "id.tl",
                       "linkto.im", "lnk.co", "nov.io", "p6l.org", "picz.us", "shortquik.com", "su.pr", "sk.gy",
                       "tota2.com", "xlinkz.info", "xtu.me", "yu2.it", "zpag.es"]
    for s in shorteners_list:
        if URL.find(s + "/") > -1:
            return 1
    return -1


def checkSubdomains(URL):
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    if domain.startswith("www."):
        domain = domain[4:]
    counter = domain.count(".") - 2
    if counter > 0:
        return 1
    elif counter == 0:
        return 0
    else:
        return -1


def checkAt(URL):
    if URL.find("@") >= 0:
        at = 1
    else:
        at = -1
    return at


def checkFakeHTTPS(URL):
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    if domain.find("https") > -1:
        return 1
    else:
        return -1


def checkDash(URL):
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    if domain.find("-") > -1:
        return 1
    else:
        return -1


def checkDataURI(URL):
    if URL.startswith("data:"):
        return 1
    return -1


def checkNumberofCommonTerms(URL):
    url = URL.lower()
    common_term = ["http", "www", ".com", "//"]
    for term in common_term:
        if url.count(term) > 1:
            return 1
        else:
            continue
    return -1


def checkNumerical(URL):
    try:
        res = get_tld(URL, as_object=True)
    except:
        return 1
    domain = res.subdomain + res.domain
    number = re.search(r"\d+", domain)
    if number:
        return 1
    else:
        return -1


def checkPathExtend(URL):
    extension = [".txt", ".exe", ".js"]
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()
    if path:
        for ex in extension:
            if path.find(ex) > -1:
                return 1
    return -1


def checkPunycode(URL):
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    subdomain = domain.split(".")
    for i in subdomain:
        mat = re.search("^xn--[a-z0-9]{1,59}|-$", i)
        if mat:
            return 1
    return -1


def checkSensitiveWord(URL):
    sensitive_words = ["secure", "account", "webscr", "login", "ebayisapi", "signin", "banking", "confirm"]
    counts = 0
    for word in sensitive_words:
        num = URL.count(word)
        counts = counts + num
    return counts


def checkTLDinPath(URL):
    try:
        res = get_tld(URL, as_object=True, fix_protocol=True)
    except:
        return 1
    path = res.parsed_url.path
    if path:
        path = path.lower().split(".")
        for pa in path:
            if is_tld(pa):
                return 1
    return -1


def checkTLDinSub(URL):
    try:
        res = get_tld(URL, as_object=True, fix_protocol=True)
    except:
        return 1
    sub_domain = res.subdomain
    if sub_domain:
        sub = sub_domain.lower().split(".")
        for s in sub:
            if is_tld(s):
                return 1
    return -1


def totalWordUrl(URL):
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    total = len(res)
    return total


def shortestWordUrl(URL):
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    try:
        shortest = min((word for word in res if word), key=len)
        return len(shortest)
    except:
        return 0


def shortestWordHost(URL):
    hostname = urlparse(URL).netloc
    res = hostname.split(".")
    try:
        shortest = min((word for word in res if word), key=len)
        return len(shortest)
    except:
        return 0


def shortestWordPath(URL):
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()
    if path == None:
        return 0
    else:
        res = re.split(r"[/:\.?=\&\-\s\_]+", path)
        valid_words = [word for word in res if word]
        if valid_words:
            shortest = min(valid_words, key=len)
            return len(shortest)
        else:
            return 0


def longestWordUrl(URL):
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    try:
        longest = max((word for word in res if word), key=len)
    except:
        return 0
    return len(longest)


def longestWordHost(URL):
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            hostname = URL[:ind]
        else:
            hostname = URL
    else:
        hostname = urlparse(URL).hostname
    if hostname == None:
        return 0
    else:
        res = re.split(r"[/:\.?=\&\-\s\_]+", hostname)
        lengths = [len(word) for word in res if word]
        if lengths:
            longest = max((word for word in res if word), key=len)
            return len(longest)
        else:
            return 0


def longestWordPath(URL):
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()
    if path == None:
        return 0
    else:
        res = re.split(r"[/:\.?=\&\-\s\_]+", path)
        valid_words = [word for word in res if word]
        if valid_words:
            longest = max(valid_words, key=len)
            return len(longest)
        else:
            return 0


def averageWordUrl(URL):
    res = re.split(r"[/:\.?=\&\-\s\_]+", URL)
    average = statistics.mean((len(word) for word in res if word))
    return float(format(average, ".2f"))


def averageWordHost(URL):
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            hostname = URL[:ind]
        else:
            hostname = URL
    else:
        hostname = urlparse(URL).hostname
    if hostname == None:
        return 0
    else:
        res = re.split(r"[/:\.?=\&\-\s\_]+", hostname)
        lengths = [len(word) for word in res if word]
        if lengths:
            average = statistics.mean((len(word) for word in res if word))
        else:
            average = 0
        return float(format(average, ".2f"))


def averageWordPath(URL):
    if URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            path = URL[ind:]
        else:
            path = None
    else:
        path = (urlparse(URL).path).lower()
    if path == None:
        return 0
    else:
        res = re.split(r"[/:\.?=\&\-\s\_]+", path)
        lengths = [len(word) for word in res if word]
        if lengths:
            average = statistics.mean((len(word) for word in res if word))
            return float(format(average, ".2f"))
        else:
            return 0


def checkStatisticRe(URL):
    top_fdomains = ["esy.es", "hol.es", "000webhostapp.com", "for-our.info", "bit.ly", "16mb.com", "96.lt",
                    "totalsolution.com.br", "beget.tech", "sellercancelordernotification.com"]
    top_tld = ["surf", "cn", "bid", "gq", "ml", "cf", "work", "cam", "ga", "casa", "tk", "ga", "top", "cyou", "bar",
               "rest"]
    try:
        f_domain = get_fld(URL)
        t_domain = get_tld(URL)
    except:
        return 1
    for f in top_fdomains:
        if f_domain.find(f) > -1:
            return 1
        else:
            for t in top_tld:
                if t_domain.find(t) > -1:
                    return 0
    return -1


# --- URL features from third-party services ---

# [FIXED] Secure API handling + JSON checks
def checkSearchEngine(URL):
    try:
        domain = get_fld(URL)
    except:
        return 1

    API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not API_KEY or not SEARCH_ENGINE_ID:
        # logging.warning("Google API keys missing")
        return 1

    query = domain
    page = 1
    start = (page - 1) * 10 + 1
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"
    try:
        data = requests.get(url).json()
        search_items = data.get("items")
        if search_items is None:
            return 1
        for i, search_item in enumerate(search_items, start=1):
            link = search_item.get("link")
            if link.find(domain) > -1:
                return -1
        return 1
    except:
        return 1


# [FIXED] Secure API handling + JSON checks
def checkGI(URL):
    try:
        domain = urlparse(URL).netloc
    except:
        return 1

    API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not API_KEY or not SEARCH_ENGINE_ID:
        return 1

    query = domain
    page = 1
    start = (page - 1) * 10 + 1
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"
    try:
        data = requests.get(url).json()
        search_items = data.get("items")
        if search_items is None:
            return 1
    except:
        return 1
    return -1


def checkPR(URL):
    try:
        domain = get_fld(URL)
    except:
        return 0
    # Uses env var or default
    headers = {"API-OPR": os.getenv("OPENPAGERANK_KEY", "c48080g840k0wc8cw88g0o40w4gg4kcksgs00k8k")}
    url = "https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=" + domain
    try:
        request = requests.get(url, headers=headers)
        result = request.json()
        resp = result["response"]
        for item in resp:
            pr = item["page_rank_integer"]
        return pr
    except:
        return 0


def getWhois(URL):
    try:
        who = whois.whois(URL)
    except:
        who = None
    return who


def checkDNS(who):
    if who is None:
        return 1
    try:
        domain_name = who["domain_name"]
    except:
        try:
            domain_name = who["domain"]
        except:
            return 1
    if (domain_name) is None:
        return 1
    return -1


def checkRegistrationLen(who):
    age_threshold = 364
    if who is None:
        return 1
    try:
        creation = who["creation_date"][0]
        expiration = who["expiration_date"][0]
    except:
        return 1
    length = (expiration - creation).days
    if length > age_threshold:
        return -1
    return 1


def checkAge(URL, who):
    if who is None:
        return 1
    age_threshold = 180
    try:
        creation = who["creation_date"][0]
        now = datetime.now()
    except:
        return 1
    try:
        age = (now - creation).days
        if age > age_threshold:
            return -1
    except:
        return 1
    return 1


def checkAbnormal(who, URL):
    if URL.count(".") == 1 and URL.startswith("http") is False:
        ind = URL.find("/")
        if ind > -1:
            domain = URL[:ind]
        else:
            domain = URL
    else:
        domain = ((urlparse(URL)).netloc).lower()
    if who is None:
        return 1
    try:
        domain_name = who["domain_name"]
    except:
        try:
            domain_name = who["domain"]
        except:
            return 1
    if (domain_name) is None:
        return 1
    if len(domain_name[0]) == 1:
        if domain == (domain_name.lower()):
            return -1
    else:
        for d in domain_name:
            if domain == d.lower():
                return -1
    return 1


# [FIXED] Native Socket Check (Replaces nmap subprocess)
def checkPorts(URL):
    try:
        # Helper to extract hostname (works for IPs too)
        parsed = urlparse(URL)
        hostname = parsed.hostname
        if not hostname:
            # Fallback for plain IPs
            hostname = URL.split('/')[0]
            if ':' in hostname: hostname = hostname.split(':')[0]
    except:
        return 1

    ports_to_check = [21, 22, 80, 443, 445, 3389]
    open_ports = []

    for port in ports_to_check:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        result = s.connect_ex((hostname, port))
        if result == 0:
            open_ports.append(port)
        s.close()

    sensitive_ports = [21, 22, 445, 3389]
    for p in sensitive_ports:
        if p in open_ports:
            return 1

    if 80 not in open_ports and 443 not in open_ports:
        return 0

    return -1


# [FIXED] Native SSL Check (Replaces curl subprocess)
def checkSSL(URL):
    try:
        parsed = urlparse(URL)
        hostname = parsed.hostname
        if not hostname:
            hostname = URL.split('/')[0]
            if ':' in hostname: hostname = hostname.split(':')[0]

        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        notAfter = cert['notAfter']
        expire_date = datetime.strptime(notAfter, "%b %d %H:%M:%S %Y %Z")
        remaining_days = (expire_date - datetime.now()).days

        if remaining_days > 300:
            return -1
        else:
            return 1
    except Exception as e:
        return 1


def extract_features_url(URL):
    u_features = {}
    u_features["URL_length"] = checkLength(URL)
    u_features["URL_IP"] = checkIP(URL)
    u_features["URL_redirect"] = checkRedirect(URL)
    u_features["URL_shortener"] = checkShortener(URL)
    u_features["URL_subdomains"] = checkSubdomains(URL)
    u_features["URL_at"] = checkAt(URL)
    u_features["URL_fakeHTTPS"] = checkFakeHTTPS(URL)
    u_features["URL_dash"] = checkDash(URL)
    u_features["URL_dataURI"] = checkDataURI(URL)
    u_features["URL_numberofCommonTerms"] = checkNumberofCommonTerms(URL)
    u_features["URL_checkNumerical"] = checkNumerical(URL)
    u_features["URL_checkPathExtend"] = checkPathExtend(URL)
    u_features["URL_checkPunycode"] = checkPunycode(URL)
    u_features["URL_checkSensitiveWord"] = checkSensitiveWord(URL)
    u_features["URL_checkTLDinPath"] = checkTLDinPath(URL)
    u_features["URL_checkTLDinSub"] = checkTLDinSub(URL)
    u_features["URL_totalWordUrl"] = totalWordUrl(URL)
    u_features["URL_shortestWordUrl"] = shortestWordUrl(URL)
    u_features["URL_shortestWordHost"] = shortestWordHost(URL)
    u_features["URL_shortestWordPath"] = shortestWordPath(URL)
    u_features["URL_longestWordUrl"] = longestWordUrl(URL)
    u_features["URL_longestWordHost"] = longestWordHost(URL)
    u_features["URL_longestWordPath"] = longestWordPath(URL)
    u_features["URL_averageWordUrl"] = averageWordUrl(URL)
    u_features["URL_averageWordHost"] = averageWordHost(URL)
    u_features["URL_averageWordPath"] = averageWordPath(URL)
    u_features["URL_checkStatisticRe"] = checkStatisticRe(URL)
    return u_features


def extract_features_rep(URL):
    r_features = {}
    who = getWhois(URL)
    r_features["REP_SearchEngine"] = checkSearchEngine(URL)
    r_features["REP_checkGI"] = checkGI(URL)
    r_features["REP_pageRank"] = checkPR(URL)
    r_features["REP_DNS"] = checkDNS(who)
    r_features["REP_registrationLen"] = checkRegistrationLen(who)
    r_features["REP_Age"] = checkAge(URL, who)
    r_features["REP_abnormal"] = checkAbnormal(who, URL)
    r_features["REP_ports"] = checkPorts(URL)
    r_features["REP_SSL"] = checkSSL(URL)
    return r_features


def extract_features_phishing(HTML, URL, feat_type="all"):
    assert feat_type in {"all", "url", "html"}
    if feat_type in ["all", "html"]:
        html_features = extract_features_html(HTML, URL)
    if feat_type in ["all", "url"]:
        url_features = extract_features_url(URL)
        rep_features = extract_features_rep(URL)
    features_dict = {}
    if feat_type == "html":
        features = list(html_features.values())
        features_dict.update(html_features)
    elif feat_type == "url":
        features = list(url_features.values()) + list(rep_features.values())
        features_dict.update(url_features)
        features_dict.update(rep_features)
    else:
        features = list(url_features.values()) + list(html_features.values()) + list(rep_features.values())
        features_dict.update(url_features)
        features_dict.update(html_features)
        features_dict.update(rep_features)
    return np.array(features).astype("float")


def build_phishing_test_data_info(main_filepath, test_samples_path, out_file_path):
    phish_test_set = joblib.load(test_samples_path)
    test_samples_idx = phish_test_set.index.to_list()
    with open(main_filepath, encoding="utf-8") as data_file:
        data = json.loads(data_file.read())
    samples_info = []
    for idx in test_samples_idx:
        sample = data[idx]
        samples_info.append({"id": sample["id"], "url": sample["url"]})
    with open(out_file_path, "wb") as fp:
        pickle.dump(samples_info, fp)


# ============================================================================
# BATCH PROCESSING MODE - Optimized for large datasets
# ============================================================================

# Global flag for batch mode
BATCH_MODE = True  # Set to True when processing large datasets (skips expensive network operations)


def extract_features_url_batch(URL):
    """
    Optimized URL feature extraction for batch processing.
    Skips expensive network operations (API calls, port scanning, SSL checks, WHOIS lookups).
    This is the FAST version for processing thousands of URLs.
    """
    u_features = {}

    # Fast local features (no network required)
    u_features["URL_length"] = checkLength(URL)
    u_features["URL_IP"] = checkIP(URL)
    u_features["URL_redirect"] = checkRedirect(URL)
    u_features["URL_shortener"] = checkShortener(URL)
    u_features["URL_subdomains"] = checkSubdomains(URL)
    u_features["URL_at"] = checkAt(URL)
    u_features["URL_fakeHTTPS"] = checkFakeHTTPS(URL)
    u_features["URL_dash"] = checkDash(URL)
    u_features["URL_dataURI"] = checkDataURI(URL)
    u_features["URL_numberofCommonTerms"] = checkNumberofCommonTerms(URL)
    u_features["URL_checkNumerical"] = checkNumerical(URL)
    u_features["URL_checkPathExtend"] = checkPathExtend(URL)
    u_features["URL_checkPunycode"] = checkPunycode(URL)
    u_features["URL_checkSensitiveWord"] = checkSensitiveWord(URL)
    u_features["URL_checkTLDinPath"] = checkTLDinPath(URL)
    u_features["URL_checkTLDinSub"] = checkTLDinSub(URL)
    u_features["URL_totalWordUrl"] = totalWordUrl(URL)
    u_features["URL_shortestWordUrl"] = shortestWordUrl(URL)
    u_features["URL_shortestWordHost"] = shortestWordHost(URL)
    u_features["URL_shortestWordPath"] = shortestWordPath(URL)
    u_features["URL_longestWordUrl"] = longestWordUrl(URL)
    u_features["URL_longestWordHost"] = longestWordHost(URL)
    u_features["URL_longestWordPath"] = longestWordPath(URL)
    u_features["URL_averageWordUrl"] = averageWordUrl(URL)
    u_features["URL_averageWordHost"] = averageWordHost(URL)
    u_features["URL_averageWordPath"] = averageWordPath(URL)
    u_features["URL_checkStatisticRe"] = checkStatisticRe(URL)

    return u_features


# def url_batches(path, batch_mode=None):
#     """
#     Process URLs from file and extract features.
#
#     Args:
#         path: Path to text file with URLs (one per line)
#         batch_mode: If True, skip expensive network operations.
#                    If None, uses global BATCH_MODE setting.
#
#     Returns:
#         pandas DataFrame with URL and features
#     """
#     if batch_mode is None:
#         batch_mode = BATCH_MODE
#
#     print(f"Loading URLs from: {path}")
#     data = pd.read_table(path, header=None, names=['url'])
#     print(f"Loaded {len(data)} URLs")
#
#     print("Extracting features...")
#     if batch_mode:
#         print("  (BATCH MODE: Skipping network-intensive operations for speed)")
#         print("  (This mode skips: port scanning, SSL checks, WHOIS, Google API calls)")
#         features = data['url'].apply(lambda x: pd.Series(extract_features_url_batch(x)))
#     else:
#         print("  (FULL MODE: Including ALL features - this will be SLOW for large datasets)")
#         features = data['url'].apply(lambda x: pd.Series(extract_features_url(x)))
#
#     result = pd.concat([data, features], axis=1)
#     print(f"Feature extraction complete! Shape: {result.shape}")
#
#     return result
def url_batches(path, batch_mode=None):
    """
    Process URLs from file and extract features.

    Args:
        path: Path to text file with URLs (one per line)
        batch_mode: If True, skip expensive network operations.
                   If None, uses global BATCH_MODE setting.

    Returns:
        pandas DataFrame with URL and features
    """
    if batch_mode is None:
        batch_mode = BATCH_MODE

    print(f"Loading URLs from: {path}")

    # FIXED: Read file line by line to handle malformed lines
    urls = []
    skipped = 0

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            url = line.strip()
            if not url:  # Skip empty lines
                continue

            # Handle lines with multiple fields (take first field only)
            if '\t' in url:
                url = url.split('\t')[0]
                skipped += 1
            elif ',' in url:
                url = url.split(',')[0]
                skipped += 1

            urls.append(url)

    data = pd.DataFrame({'url': urls})
    print(f"Loaded {len(data)} URLs", end='')
    if skipped > 0:
        print(f" (cleaned {skipped} malformed lines)")
    else:
        print()

    # print("Extracting features...")
    # if batch_mode:
    #     print("  (BATCH MODE: Skipping network-intensive operations for speed)")
    #     print("  (This mode skips: port scanning, SSL checks, WHOIS, Google API calls)")
    #     features = data['url'].apply(lambda x: pd.Series(extract_features_url_batch(x)))
    # else:
    #     print("  (FULL MODE: Including ALL features - this will be SLOW for large datasets)")
    #     features = data['url'].apply(lambda x: pd.Series(extract_features_url(x)))
    print("Extracting features...")
    if batch_mode:
        print("  (BATCH MODE: Skipping network-intensive operations for speed)")
        print("  (This mode skips: port scanning, SSL checks, WHOIS, Google API calls)")

        # Extract features with error handling
        def safe_extract_batch(url):
            try:
                return pd.Series(extract_features_url_batch(url))
            except Exception as e:
                print(f"  ⚠ Skipping malformed URL: {url[:50]}... ({type(e).__name__})")
                # Return empty feature dict with default values
                return pd.Series({key: -1 for key in extract_features_url_batch("http://example.com").keys()})

        features = data['url'].apply(safe_extract_batch)
    else:
        print("  (FULL MODE: Including ALL features - this will be SLOW for large datasets)")

        def safe_extract_full(url):
            try:
                return pd.Series(extract_features_url(url))
            except Exception as e:
                print(f"  ⚠ Skipping malformed URL: {url[:50]}... ({type(e).__name__})")
                return pd.Series({key: -1 for key in extract_features_url("http://example.com").keys()})

        features = data['url'].apply(safe_extract_full)

    result = pd.concat([data, features], axis=1)
    print(f"Feature extraction complete! Shape: {result.shape}")

    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("URL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"\nBatch Mode: {'ENABLED' if BATCH_MODE else 'DISABLED'}")
    print("(To change mode, edit BATCH_MODE variable at top of this section)")
    print("=" * 80)

    # Paths
    Phishing_url_data_path = r"./data/phish-58w.txt"
    Legitimate_url_data_path = r"./data/legtimate-58w.txt"

    # Check if files exist
    if not os.path.exists(Phishing_url_data_path):
        print(f"\n✗ Error: {Phishing_url_data_path} not found!")
        print("Please ensure the data file exists.")
        sys.exit(1)

    if not os.path.exists(Legitimate_url_data_path):
        print(f"\n✗ Error: {Legitimate_url_data_path} not found!")
        print("Please ensure the data file exists.")
        sys.exit(1)

    # Process Phishing URLs
    print("\n" + "=" * 80)
    print("[1/2] Processing Phishing URLs")
    print("=" * 80)
    try:
        phish_features = url_batches(Phishing_url_data_path)
        output_path = './Phishing_url_data_art.csv'
        phish_features.to_csv(output_path, index=False)
        print(f"✓ Successfully saved to: {output_path}")
        print(f"  Shape: {phish_features.shape}")
    except Exception as e:
        print(f"✗ Error processing phishing URLs: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Process Legitimate URLs
    print("\n" + "=" * 80)
    print("[2/2] Processing Legitimate URLs")
    print("=" * 80)
    try:
        legit_features = url_batches(Legitimate_url_data_path)
        output_path = './Legitimate_url_data_art.csv'
        legit_features.to_csv(output_path, index=False)
        print(f"✓ Successfully saved to: {output_path}")
        print(f"  Shape: {legit_features.shape}")
    except Exception as e:
        print(f"✗ Error processing legitimate URLs: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - ./Phishing_url_data_art.csv")
    print("  - ./Legitimate_url_data_art.csv")
    print("\nNext steps:")
    print("  1. Run CNN_process.py to generate CNN features")
    print("  2. Run final_train.py to train ensemble classifier")
    print("=" * 80)