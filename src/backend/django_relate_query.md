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

可以使用prefetch对象来控制prefetch的查询

可以通过to_attr参数分配查询的结果到一个指定的属性,结果直接保存在列表中 