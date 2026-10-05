from django import template

register = template.Library()


@register.filter
def lookup(mapping, key):
    if mapping is None:
        return None
    return mapping.get(key)
