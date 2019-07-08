## 1. mysql autocommit

> In InnoDB, all user activity occurs inside a transaction. If autocommit mode is enabled, each SQL statement forms a single transaction on its own. By default, MySQL starts the session for each new connection with autocommit enabled, so MySQL does a commit after each SQL statement if that statement did not return an error. If a statement returns an error, the commit or rollback behavior depends on the error. 


## 2. Why Django uses autocommit¶

> In the SQL standards, each SQL query starts a transaction, unless one is already active. Such transactions must then be explicitly committed or rolled back.

> This isn’t always convenient for application developers. To alleviate this problem, most databases provide an autocommit mode. When autocommit is turned on and no transaction is active, each SQL query gets wrapped in its own transaction. In other words, not only does each such query start a transaction, but the transaction also gets automatically committed or rolled back, depending on whether the query succeeded.

>PEP 249, the Python Database API Specification v2.0, requires autocommit to be initially turned off. Django overrides this default and turns autocommit on.


## 3. django's default transaction behavior

> django 默认 autocommit mode , 即每个语句立即commit到数据库, 除非激活了事务.

## 4. transactions to http requests

> 设置 ATOMIC_REQUESTS=True , 在调用view函数之前 django开启事务 ,如果无错误返回响应django提交,相反则回滚

> 当 ATOMIC_REQUESTS=True @transaction.non_atomic_requests 装饰器的view不会开启事务

## 5. controlling transactions explicitly

> atomic 可以确保了一段代码在数据库的原子性, 
> atomic 可以嵌套, 即使内层已经提交, 在外层出错的情况下内层仍可回滚.
> 
> atomic use both as a decorator and as a context manager

```python
from django.db import transaction

@transaction.atomic
def view(request):
    # inside transactions
    pass

def view_2(request):
    # in autocommit mode
    with transaction.atomic():
        # inside transactions
        pass
```

> 避免在atomimc内部捕捉错误

> 当事务回滚时,对应model的字段不会被恢复

> 为确保原子性, atomic block 内, 尝试提交,回滚,修改autocommit状态的的都是出错

## 6. commit 之后的操作

> 一些时候需要在数据库事务提交成功后做一些事, django 提供on_commit()函数 来注册一个callback function, 当事务提交后立即执行

```python
from django.db import transaction

def do_sth():
    pass

def view(request):
    # in autocommit mode
    with transaction.atomic():
        transaction.on_commit(do_sth)
        # inside transactions
        pass
```

## 7. save point

> savepoint 是一个可以让你回滚部分事务的标记,支持sqlite, postgresql,oracle,mysql
> 一但你使用atomic()开启了事务,一系列的数据库操作一同执行或回滚,savepoints提供了良好的回滚组织,而不是像transaction.roolback()所有操作一同回滚
> 当atomic()嵌套,会创建一个savepoint允许部分提交或回滚

> savepoint通过django.db.transaction 中的三个函数控制
>
> > savepoint(using=None): 创建savepoint ,return savepoint ID
>
> > avepoint_commit(sid,using=None): 释放一个sid 保留改动
> 
> > savepoint_rollback(sid,using=None): 回滚

```python
with transaction.atomic():  # Outer atomic, start a new transaction
    transaction.on_commit(foo)

    with transaction.atomic():  # Inner atomic block, create a savepoint
        transaction.on_commit(bar)

# foo() and then bar() will be called when leaving the outermost block

#====================================

with transaction.atomic():  # Outer atomic, start a new transaction
    transaction.on_commit(foo)

    try:
        with transaction.atomic():  # Inner atomic block, create a savepoint
            transaction.on_commit(bar)
            raise SomeError()  # Raising an exception - abort the savepoint
    except SomeError:
        pass

# foo() will be called, but not bar()
```