# -*- coding: utf-8 -*-

class StringAlgorithm:
    def valid_ip_address(self, query_ip: str) -> str:
        """验证IP地址
        (leetcode 468) 给定一个字符串，如果符合IPv4格式，返回"IPv4"；如果符合IPv6格式，返回"IPv6"；否则返回"Neither"。
        思路：如果query中存在 . 按照ipv4判断; 如果query中存在 : 按ipv6判断; 否则都不是。
        ipv4：1) len=4 -> 2) x是数字；3) 0<=x<=255; 4) 长度大于1时，第一位不为0
        ipv6：1) len=8 -> 2) 1<=x.len()<=4; 3) x[i]是数字，或x[i] in [a-fA-F]
        时O(n); 空O(1)
        """
        if '.' in query_ip:
            ip_list = query_ip.split('.')
            if len(ip_list) != 4:
                return "Neither"
            for ip in ip_list:
                if not ip.isdigit() or int(ip) < 0 or int(ip) > 255 or (
                        len(ip) > 1 and int(ip[0]) == 0):
                    return "Neither"

            return "IPv4"

        elif ':' in query_ip:
            ip_list = query_ip.split(':')
            if len(ip_list) != 8:
                return "Neither"
            for ip in ip_list:
                if len(ip) < 1 or len(ip) > 4:
                    return "Neither"
                else:
                    s = ['a', 'b', 'c', 'd', 'e', 'f',
                         'A', 'B', 'C', 'D', 'E', 'F']
                    for i in ip:
                        if not i.isdigit() and i not in s:
                            return "Neither"
            return "IPv6"

        else:
            return "Neither"

# 最大无重复字串长度