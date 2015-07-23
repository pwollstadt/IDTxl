import types

# example taken from:
# http://stackoverflow.com/questions/962962/python-changing-methods-and-attributes-at-runtime

class SpecialClass(object):
    @classmethod
    def removeVariable(cls, name):
        return delattr(cls, name)
    @classmethod
    def addMethod(cls, func):
        return setattr(cls, func.__name__, types.MethodType(func, cls))
    @classmethod
    def addMethodAs(cls, func, new_name):
        return setattr(cls, new_name, types.MethodType(func, cls))

def hello(self, n):
    print(n)

instance = SpecialClass()
SpecialClass.addMethod(hello)
SpecialClass.hello(5)
instance.hello(6)
inst.addMethodAs(hello, "quack")
inst.quack(5)
SpecialClass.removeVariable("hello")
instance.hello(7)
SpecialClass.hello(8)
