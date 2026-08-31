// pybind11 module: nanoserve_core. One class for now (the block allocator);
// the scheduler step lands here in C2 as a single fat schedule_step() call so
// the Python/C++ boundary is crossed once per step, not once per operation.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "allocator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nanoserve_core, m) {
    m.doc() = "nanoserve C++ hot path: block allocator (C1)";

    py::class_<nano::BlockAllocator>(m, "BlockAllocator")
        .def(py::init<uint32_t, uint32_t, bool>(),
             py::arg("num_blocks"), py::arg("block_size"),
             py::arg("lockfree") = false)
        .def_property_readonly("num_blocks", &nano::BlockAllocator::num_blocks)
        .def_property_readonly("block_size", &nano::BlockAllocator::block_size)
        .def_property_readonly("num_free", &nano::BlockAllocator::num_free)
        .def_property_readonly("num_used", &nano::BlockAllocator::num_used)
        .def("can_admit", &nano::BlockAllocator::can_admit, py::arg("n_tokens"))
        .def("add_seq", &nano::BlockAllocator::add_seq,
             py::arg("sid"), py::arg("n_tokens"))
        .def("append_token",
             [](nano::BlockAllocator& a, int64_t sid) -> py::object {
                 uint32_t b = a.append_token(sid);
                 if (b == nano::kNull) return py::none();
                 return py::int_(b);
             },
             py::arg("sid"))
        .def("free_seq", &nano::BlockAllocator::free_seq, py::arg("sid"))
        .def("table", &nano::BlockAllocator::table, py::arg("sid"),
             py::return_value_policy::copy)
        .def("seq_length", &nano::BlockAllocator::seq_length, py::arg("sid"))
        .def_property_readonly("num_seqs", &nano::BlockAllocator::num_seqs)
        // batch op: grow many sequences in one crossing (the C2 direction)
        .def("append_tokens",
             [](nano::BlockAllocator& a, const std::vector<int64_t>& sids) {
                 std::vector<int64_t> new_blocks;
                 new_blocks.reserve(sids.size());
                 {
                     py::gil_scoped_release release;
                     for (int64_t sid : sids) {
                         uint32_t b = a.append_token(sid);
                         new_blocks.push_back(
                             b == nano::kNull ? -1 : int64_t(b));
                     }
                 }
                 return new_blocks;
             },
             py::arg("sids"));
}
