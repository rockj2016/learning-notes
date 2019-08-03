# prefetch_related()

在一个batch内,返回一个自动获取queryset,包含指定查询的相关对象

和select_related一样都是为了解决因访问关联对象而频繁查询数据库的问题

select_related工作原理是在 select 查询中使用join, 所以select_related在一次查询中可以获取所有数据, 为了避免因为出现大量的查询结果, select_related 只能在single-valued relationships - foreign key and one-to-one中使用

prefetch_related 单独查询表, 把数据在python中再关联起来, 所以可以预查询 一对多,多对多

prefetch_related()中额外附加的查询会在主查寻结束后,queryset开始处理时进行

主要查询和指定的关联对象都会被全部加载到内存中,这违背了queryset宗旨--
避免在需要使用之前(甚至查询在数据库执行之后)把所有数据加载到内存中

> 注意, 任何queryset上的链式查询实施了不同的查询,则之前的缓存结果会被忽略,获取到的数据直接使用新的查询

```python
pizzas = Pizza.objects.prefetch_related('toppings')
[list(pizza.toppings.filter(spicy=True)) for pizza in pizzas]
```

> 代码中, pizza.toppings.all()为预查询数据,prefetch_related('toppings') 的实现是pizza.toppings.all(), 而pizza.toppings.filter()是一个全新不同的查询,预获取的缓存数据在这种情况下用不上
> 如果在related_managers中使用了更改数据库的方法,add(),remove(),clear(),set(),缓存的关系数据会被清空

还可以对外键的外键进行查询

```python
class Restaurant(models.Model):
    pizzas = models.ManyToManyField(Pizza, related_name='restaurants')
    best_pizza = models.ForeignKey(Pizza, related_name='championed_by', on_delete=models.CASCADE)
# 3query
Restaurant.objects.prefetch_related('pizzas__toppings')

# 3query
Restaurant.objects.prefetch_related('best_pizza__toppings')

# 2query
Restaurant.objects.select_related('best_pizza').prefetch_related('best_pizza__toppings')
```

可以使用select_related 简化第二个查询, 由于prefetch 会在主查询(包含了select_relate)之后执行, best_pizza数据已经获取了,所以会在prefetch时会跳过这部分

prefetch_related调用会累积很多数据, 可以传入None来清除prefetch_related行为

```python
non_prefetched = qs.prefetch_related(None)
```

prefetch_related在多数情况下在sql中使用in运算符, 一个很多数据的queryset会产生很长的in语句,在sql的解析或执行上可能有性能问题

> 如果使用迭代器来遍历queryset, prefetch_related会被忽略

可以使用Prefetch对象来控制prefetch的查询, 最简单得情况下Prefetch 和 使用 prefect_related直接传入字段名称一样

```python
from django.db.models import Prefetch
Restaurant.objects.prefetch_related(Prefetch('pizzas__toppings'))
```

也可以通过queryset 参数提供一个自定义queryset, 意义在于过滤,或自定义排序之类
或者是在prefetch中 调用select_related 进一步减少查询次数

```python
queryset = Toppings.objects.order_by('name')
Restaurant.objects.prefetch_related(
Prefetch('pizzas__toppings', queryset=queryset))

queryset = Restaurant.objects.select_related('best_pizza')
Pizza.objects.prefetch_related(
Prefetch('restaurants', queryset=queryset))
```

你也可用通过to_attr 将prefetched 结果指定到一个自定义属性上,结果会直接保存在一个list里面, 使用to_attr 可以对一个属性使用不同条件多次查询

```python
vegetarian_pizzas = Pizza.objects.filter(vegetarian=True)
Restaurant.objects.prefetch_related(
    Prefetch('pizzas', to_attr='menu'),
    Prefetch('pizzas', queryset=vegetarian_pizzas, to_attr='vegetarian_menu'))
```

在对prefect结果进行过滤的情况下,建议使用to_attr,避免歧义

```python
queryset = Pizza.objects.filter(vegetarian=True)

# Recommended:
restaurants = Restaurant.objects.prefetch_related(
     Prefetch('pizzas', queryset=queryset, to_attr='vegetarian_pizzas'))
vegetarian_pizzas = restaurants[0].vegetarian_pizzas
 # Not recommended:
restaurants = Restaurant.objects.prefetch_related(
    Prefetch('pizzas', queryset=queryset))
vegetarian_pizzas = restaurants[0].pizzas.all()
```

自定义的 prefetching 同样也适用于 单一关系查询: 正向的ForeignKey, OneToOnewField
,上面二者通常情况下可以使用 select_related(), 但是一些情况下 prefetching 一个自定义的QuserSet更好用

- 需要对关联对象进一步prefetch
- 只需要 prefetch 关联对象的子集
- 使用查询优化技巧,如 deferred fields

---

# Prefetch() objects

## class Prefetch(lookup, queryset=None, to_attr=None)

Prefetch 对象用于控制 prefetch_related() 查询

lookup参数是查询的关系字段, 和传递给prefetch_related()的基于字符串的查询一样

queryset 对于查询的field提供一个基础的查询, 当要对prefetch结果进一步过滤时, 或者对prefetch的字段再调用select_related()时, 就很重要了.

```python
voted_choices = Choice.objects.filter(votes__gt=0)
voted_choices
<QuerySet [<Choice: The sky>]>
prefetch = Prefetch('choice_set', queryset=voted_choices)
Question.objects.prefetch_related(prefetch).get().choice_set.all()
<QuerySet [<Choice: The sky>]>
```

to_attr 参数设置prefetch结果到自定义属性上

```python
prefetch = Prefetch('choice_set', queryset=voted_choices, to_attr='voted_choices')
Question.objects.prefetch_related(prefetch).get().voted_choices
[<Choice: The sky>]
Question.objects.prefetch_related(prefetch).get().choice_set.all()
<QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>
```

> 当使用to_attr时,prefetch 结果会保存在list里面, 这比传统prefetch_related(prefetch结果保存在queryset中)快了很多,

---

# defer()

## defer(*fielfs)

在一些数据建模情况, model可能包含很多字段, 有些字段可能包含很多数据(如 text fields),将这些数据转化为 Python 对象需要很多开销, 某些情况下在你不知道一些字段是否会用上,可以在初始查询时不获取它们.

我们可以将第一时间不需要获取的字段传到defer()

```python
Entry.objects.defer("headline", "body")
```

有了deferred字段的查询还是会返回model实例, 当访问实例中deferred属性时才会从数据库获取(一次一个deferred字段)

添加到defer中字段的顺序没有影响

可以懒加载关联模型中的字段(关联模型通过select_related()加载的情况),通过使用标准的双下划线来区分关联模型中字段

```python
Blog.objects.select_related().defer("entry__headline", "entry__body")
```

如果想清空derferred字段, 就向defer()中传入None

```python
# Load all fields immediately.
my_queryset.defer(None)
```

一些字段不能懒加载, primarykey, 如果你使用select_related()去获取关联模型,这时就不应该懒加载相关字段,这样会产生错误

> defer()方法是较为特殊的情况下使用,尽量在不得不使用的情况下才使用,
> 如果你需要频繁的使用model中部分字段,最好的选择是标准化你的models,将不需要加载的字段放到另一个模型, 如果字段必须要留在表中, 可以使用Meta.managed=False创建一个模型,只包含进程会用到的字段, 这样代码更易于阅读,更快同事也消耗更少内存

```python
class CommonlyUsedModel(models.Model):
    f1 = models.CharField(max_length=10)

    class Meta:
        managed = False
        db_table = 'app_largetable'

class ManagedModel(models.Model):
    f1 = models.CharField(max_length=10)
    f2 = models.CharField(max_length=10)

    class Meta:
        db_table = 'app_largetable'

# Two equivalent QuerySets:
CommonlyUsedModel.objects.all()
ManagedModel.objects.all().defer('f2')
```

如果在unmanaged model中很多字段是重复的,最好的是提取一个公共model, unmanaged 和 managed 继承公共模型

> 当对model 实例调用save()方法时, 只有非懒加载字段会被保存

---

# using()

## using(alias)

如果你使用了多个数据库,这个方法控制QuerySet的来源数据库

```python
# queries the database with the 'default' alias.
>>> Entry.objects.all()

# queries the database with the 'backup' alias
>>> Entry.objects.using('backup')
```

---

# select_for_update()

## select_for_update(nowait=False,skip_locked=False,of=())

返回一个带锁的queryset直到事务结束,生成了 select ... for update 的sql 语句.

```python
from django.db import transaction

entries = Entry.objects.select_for_update().filter(author=request.user)
with transaction.atomic():
    for entry in entries:
        # is evaluated
        ...
```

当queryset正在杯处理时(上例中 for entery in entries), 所有的entries都会被锁,直到事务结束,期间其它的事务会被禁止修改或获取锁

通常情况下, 如果一个事务获取了一些数据的锁, 查询会被阻塞直到锁释放, 

select_for_update() 默认给所有选中的列加锁, 在select_related中指定的关联对象的列也会被加锁,可以通过 select_for_update(of=(...)) 指定要加锁的关联对象, 使用和select_related一样的字段语法,.

不能在关联对象为null的情况下使用 select_for_update()

```python
>>> Person.objects.select_related('hometown').select_for_update()
Traceback (most recent call last):
...
django.db.utils.NotSupportedError: FOR UPDATE cannot be applied to the nullable side of an outer join
```