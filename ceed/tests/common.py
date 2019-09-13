

def add_prop_watch(obj, event, watch_prop):
    assert not hasattr(obj, watch_prop)
    setattr(obj, watch_prop, 0)

    def count_changes(*largs):
        setattr(obj, watch_prop, getattr(obj, watch_prop) + 1)

    obj.fbind(event, count_changes)


async def exhaust(async_it):
    async for _ in async_it:
        pass
