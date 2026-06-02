StateId=int
Char=str|None
EPS=None

class Frag:
    def __init__(self,start:StateId,accept:StateId,) -> None:
        self.start=start
        self.accept=accept

class Builder:
    def __init__(self) -> None:
        self.transition:dict[StateId, dict[Char, set[StateId]]] = {}
        self._next_state_id=0

    def new_state(self)->StateId:
        state_id=self._next_state_id
        self._next_state_id+=1
        self.transition[state_id]={}
        return state_id
    
    def add_edge(self,frm_id:StateId,char:Char,to_id:StateId):
        self.transition[frm_id].setdefault(char,set()).add(to_id)

    def lit(self,char:str)->Frag:
        start_id=self.new_state()
        accept_id=self.new_state()
        self.add_edge(start_id,char,accept_id)
        return Frag(start_id,accept_id)
    
def eps_closure(builder:Builder,states:set[StateId])->set[StateId]:
    closure=set(states)
    stack=list(states)
    while stack:
        state=stack.pop()
        for to_id in builder.transition[state].get(EPS,()):
            if to_id not in closure:
                closure.add(to_id)
                stack.append(to_id)
    return closure



def run_nfa(builder:Builder,regex:Frag,text:str)->bool:
    current_states=eps_closure(builder,{regex.start})
    for char in text:
        next_states=set()
        for state in current_states:
            builder.transition[state].get(char,())

        current_states=eps_closure(builder,next_states)
    return regex.accept in current_states