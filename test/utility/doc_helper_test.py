from yuning_util.dev_mode import DevMode
dev_mode = DevMode('k-seq')
dev_mode.on()

from k_seq.utility import doc_helper

# TODO: DocHelper can store parameter docs


def test_DocHelper_stores_doc():
    doc = doc_helper.DocHelper(
        arg1='Docstring for arg1',
        arg2=('Docstring for arg2'),
        arg3=('type for arg3', 'Docstring for arg3')
    )
    assert doc.var_lib.shape == (3, 2)


def test_DocHelper_compose_correctly():
    doc = doc_helper.DocHelper(
        arg1='Docstring for arg1',
        arg2=('Docstring for arg2'),
        arg3=('type for arg3', 'Docstring for arg3')
    )
    output = doc.get(['arg1'])
    assert output == "    arg1: Docstring for arg1"
    output = doc.get(['arg1', 'arg3'])
    assert output == "    arg1: Docstring for arg1\n    arg3 (type for arg3): Docstring for arg3"


def test_string_strip_func_works():
    docstring = ('This is docstring\n'
                 '    with multiple lines of args:\n'
                 '<< arg1,arg2,  arg3 >>\n'
                 '<<arg1, arg3>>\n')

    doc = doc_helper.DocHelper(
        arg1='Docstring for arg1',
        arg2=('Docstring for arg2'),
        arg3=('type for arg3', 'Docstring for arg3')
    )

    split = doc.split_string(docstring)
    assert split[0] == "This is docstring\n    with multiple lines of args:\n"
    assert split[1] == ('arg1', 'arg2', 'arg3')
    assert split[2] == '\n'
    assert split[3] == ('arg1', 'arg3')


def test_DocHelper_compose_doc_decorator_works():
    doc = doc_helper.DocHelper(
        arg1='Docstring for arg1',
        arg2=('Docstring for arg2'),
        arg3=('type for arg3', 'Docstring for arg3')
    )
    
    @doc.compose("""Here is an example
<<arg1, arg2>>
Args:
<<arg2, arg3, 8>>
""")
    def target_func(arg1):
        pass

    assert target_func.__doc__ == "Here is an example\n" \
                                  "    arg1: Docstring for arg1\n" \
                                  "    arg2: Docstring for arg2\n" \
                                  "Args:\n" \
                                  "        arg2: Docstring for arg2\n" \
                                  "        arg3 (type for arg3): Docstring for arg3\n"


def test_DocHelper_compose_doc_decorator_no_effect_works():
    doc = doc_helper.DocHelper(
        arg1='Docstring for arg1',
        arg2=('Docstring for arg2'),
        arg3=('type for arg3', 'Docstring for arg3')
    )

    @doc.compose("""Here is an example of simple docstring with no argument substitution
Args:
  args1
""")
    def target_func(arg1):
        pass

    assert target_func.__doc__ == "Here is an example of simple docstring with no argument substitution\n" \
                                  "Args:\n" \
                                  "  args1\n"
