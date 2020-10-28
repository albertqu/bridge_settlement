from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()


@stringfilter
@register.filter(name='tourl')
def name_to_url(value):
    return value.replace(" ", "-").lower()
